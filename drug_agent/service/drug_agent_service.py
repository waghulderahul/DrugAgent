"""
Drug Agent Service — universal drug intelligence API for all agents.

Usage:
    from drug_agent.service import get_service, DrugQueryRequest, GeneContext, QueryType

    svc = get_service()
    resp = svc.query(DrugQueryRequest(
        disease="Breast Cancer",
        genes=[GeneContext("ERBB2", 4.25, 0.05, "up", role="pathogenic")],
    ))
    print(resp.high_priority)

    # Convenience methods for single-purpose agents
    safety = svc.get_safety_profile("trastuzumab")
    trials = svc.get_trial_evidence("trastuzumab", "breast cancer")
"""

import time
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np

from .schemas import (
    DrugQueryRequest, DrugQueryResponse, QueryType, ScoringConfig,
    DrugCandidate, DrugIdentity, TargetEvidence, TrialEvidence,
    SafetyProfile, ScoreBreakdown, ContraindicationEntry,
)
from .collection_router import CollectionRouter
from .result_aggregator import ResultAggregator
from .drug_scorer import DrugScorer, _disease_matches

logger = logging.getLogger(__name__)


class DrugAgentService:

    def __init__(self, scoring_config: Optional[ScoringConfig] = None):
        logger.info("Initializing DrugAgentService...")
        t0 = time.time()
        self.router = CollectionRouter()
        self.aggregator = ResultAggregator()
        self.scorer = DrugScorer(config=scoring_config, embedder=self.router.embedder)
        logger.info(f"DrugAgentService ready in {time.time() - t0:.1f}s")

    # ── Full Query Pipeline ──────────────────────────────────────────────────

    def query(self, request: DrugQueryRequest) -> DrugQueryResponse:
        t0 = time.time()
        if request.scoring_config:
            self.scorer.config = request.scoring_config

        try:
            handlers = {
                QueryType.FULL_RECOMMENDATION: self._full_recommendation,
                QueryType.VALIDATE_DRUG: self._validate_drug,
                QueryType.CHECK_CONTRAINDICATION: self._check_contraindication,
                QueryType.SAFETY_PROFILE: self._safety_profile,
                QueryType.DRUG_DETAILS: self._drug_details,
            }
            handler = handlers.get(request.query_type, self._full_recommendation)
            response = handler(request)
            response.metadata["duration_seconds"] = round(time.time() - t0, 2)
            return response
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return DrugQueryResponse(
                success=False, disease=request.disease,
                query_type=request.query_type.value,
                errors=[str(e)],
                metadata={"duration_seconds": round(time.time() - t0, 2)},
            )

    def _noise_threshold(self, candidate: DrugCandidate, scoring_config: ScoringConfig) -> float:
        clinical_score = candidate.score.clinical_regulatory_score if candidate.score else 0.0
        if clinical_score > scoring_config.high_clinical_score_cutoff:
            return scoring_config.high_clinical_noise_threshold
        return scoring_config.base_noise_threshold

    def _has_signal(self, candidate: DrugCandidate, scoring_config: ScoringConfig) -> bool:
        s = candidate.score
        if not s or s.composite_score < self._noise_threshold(candidate, scoring_config):
            return False
        target_any = s.target_direction_match + s.target_magnitude_match
        positives = sum(1 for v in (
            target_any, s.clinical_regulatory_score,
            s.ot_association_score, s.pathway_concordance,
        ) if v > 0)
        return positives >= 1

    def _full_recommendation(self, request: DrugQueryRequest) -> DrugQueryResponse:
        self.router.reset_query_tracking()

        # ── STAGE 0: Disease Alias Expansion ───────────────────────────
        if not request.disease_aliases:
            request.disease_aliases = self.router.get_disease_aliases(request.disease)

        # ── STAGE 1: Candidate Discovery ─────────────────────────────────
        discovery_results: Dict[str, list] = {}

        up_targets = request.get_upregulated_targets()
        down_targets = [g for g in request.get_downregulated_genes()
                        if g.role in ("protective", "therapeutic_target", "immune_modulator")]
        print(f"        [Stage 1] Discovery: {len(up_targets)} upregulated + "
              f"{len(down_targets)} downregulated targets, "
              f"{len(request.pathways)} pathways, disease='{request.disease}'")
        print(f"          UP genes: {[g.gene_symbol for g in up_targets]}")
        print(f"          DOWN genes: {[g.gene_symbol for g in down_targets]}")
        for gene in up_targets:
            key = f"target_{gene.gene_symbol}"
            discovery_results[key] = self.router.find_drugs_for_target(gene.gene_symbol)

        for gene in down_targets:
            key = f"target_{gene.gene_symbol}"
            if key not in discovery_results:
                discovery_results[key] = self.router.find_drugs_for_target(gene.gene_symbol)

        discovery_results["disease"] = self.router.find_drugs_for_disease(
            request.disease, disease_aliases=request.disease_aliases)

        # Pathway-based discovery for top pathways by significance (any direction)
        sorted_pathways = sorted(request.pathways, key=lambda p: p.fdr)
        for pw in sorted_pathways[:5]:
            key = f"pathway_{pw.pathway_name[:30]}"
            discovery_results[key] = self.router.get_pathway_drugs(pw.pathway_name, pw.key_genes)

        gene_hits = sum(len(v) for k, v in discovery_results.items() if k.startswith('target_'))
        disease_hits = len(discovery_results.get('disease', []))
        pathway_hits = sum(len(v) for k, v in discovery_results.items() if k.startswith('pathway_'))
        merged = self.aggregator.merge_candidates(discovery_results)
        print(f"          Hits: gene={gene_hits}, disease={disease_hits}, pathway={pathway_hits} → {len(merged)} unique")
        print(f"          All discovered: {[m['drug_name'] for m in merged]}")
        logger.info(f"Stage 1: {sum(len(v) for v in discovery_results.values())} raw → {len(merged)} unique candidates")

        # ── STAGE 2: Evidence Enrichment (parallel per candidate) ────────
        candidates: List[DrugCandidate] = []

        def _enrich_one(entry):
            drug_name = entry["drug_name"]
            identity = self.router.get_drug_identity(drug_name)
            targets = self.router.get_drug_targets(drug_name)
            for t in targets:
                if t.get("gene_symbol"):
                    ot_score = self.router.get_target_disease_score(t["gene_symbol"], request.disease)
                    t["ot_association_score"] = ot_score
                    t["known_effectors"] = self.router.get_functionally_related_genes(t["gene_symbol"])

            indication = self.router.get_indication_status(
                drug_name, request.disease, disease_aliases=request.disease_aliases)

            trials = self.router.get_trial_evidence(
                drug_name, request.disease,
                disease_aliases=request.disease_aliases) if request.include_trials else {}

            # ── Synonym fallback: retry weak queries with alternative names ──
            synonym_candidates = self._build_synonym_list(
                drug_name, identity, entry.get("original_names", []))
            if synonym_candidates:
                # Retry indication if no approval found
                if not indication.get("is_approved") and not indication.get("indication_text"):
                    for syn in synonym_candidates:
                        alt_ind = self.router.get_indication_status(
                            syn, request.disease, disease_aliases=request.disease_aliases)
                        if alt_ind.get("is_approved") or alt_ind.get("indication_text"):
                            indication = alt_ind
                            logger.info(f"Synonym hit: indication for '{drug_name}' via '{syn}'")
                            break
                # Retry trials if none found
                if request.include_trials and trials.get("total_trials", 0) == 0:
                    for syn in synonym_candidates:
                        alt_trials = self.router.get_trial_evidence(
                            syn, request.disease,
                            disease_aliases=request.disease_aliases)
                        if alt_trials.get("total_trials", 0) > 0:
                            trials = alt_trials
                            logger.info(f"Synonym hit: trials for '{drug_name}' via '{syn}'")
                            break
                # Retry targets if none found
                if not targets:
                    for syn in synonym_candidates:
                        alt_targets = self.router.get_drug_targets(syn)
                        if alt_targets:
                            targets = alt_targets
                            for t in targets:
                                if t.get("gene_symbol"):
                                    ot_score = self.router.get_target_disease_score(
                                        t["gene_symbol"], request.disease)
                                    t["ot_association_score"] = ot_score
                            logger.info(f"Synonym hit: targets for '{drug_name}' via '{syn}'")
                            break
                # Retry identity for ChEMBL ID if missing
                if not identity.get("chembl_id"):
                    for syn in synonym_candidates:
                        alt_id = self.router.get_drug_identity(syn)
                        if alt_id.get("chembl_id"):
                            identity["chembl_id"] = alt_id["chembl_id"]
                            for bn in alt_id.get("brand_names", []):
                                if bn and bn not in identity["brand_names"]:
                                    identity["brand_names"].append(bn)
                            logger.info(f"Synonym hit: chembl_id for '{drug_name}' via '{syn}'")
                            break

            # Propagate indication_text into identity for scorer
            identity["indication_text"] = indication.get("indication_text", "")

            # Detect precision-medicine eligibility requirements from indication text
            _ELIGIBILITY_RE = re.compile(
                r'(?:patients?\s+with|amenable\s+to|confirmed)\s+[\w\s-]+'
                r'(?:mutation|deletion|exon[- ]?\d+|variant|amplification|overexpression'
                r'|rearrangement|fusion|translocation|deficiency)'
                r'|exon[- ]?\d+[- ]skipping|gene\s+therapy',
                re.IGNORECASE,
            )
            ind_full = identity["indication_text"]
            if ind_full:
                elig_match = _ELIGIBILITY_RE.search(ind_full)
                if elig_match:
                    identity["genetic_eligibility_required"] = True
                    identity["genetic_eligibility_detail"] = elig_match.group().strip()

            # ── Preserve discovery-stage gene links ──────────────────────
            # Discovery found this drug via specific genes, but enrichment
            # re-queries targets from scratch and may lose those links.
            # Inject any missing discovery genes as stub targets so the
            # scorer can match them against the patient's gene list.
            discovery_genes = entry.get("targets", set())
            enriched_genes = {t.get("gene_symbol", "").upper() for t in targets}
            has_chembl_targets = bool(enriched_genes)
            for dg in discovery_genes:
                if dg.upper() not in enriched_genes:
                    ot_score = self.router.get_target_disease_score(dg, request.disease)
                    # If drug has ChEMBL targets but this gene isn't one of them,
                    # it's an indirect relationship (e.g., Belimumab→BAFF, not BLK)
                    inferred_action = "INDIRECT_EFFECT" if has_chembl_targets else "UNKNOWN"
                    targets.append({
                        "gene_symbol": dg,
                        "action_type": inferred_action,
                        "mechanism": None,
                        "fda_narrative": None,
                        "ot_association_score": ot_score,
                        "known_effectors": self.router.get_functionally_related_genes(dg),
                    })

            trials = trials if request.include_trials else {}
            safety = self.router.get_safety_profile(drug_name) if request.include_safety else {}

            sources = list(entry.get("sources", []))
            candidate = self.aggregator.build_candidate(
                drug_name, identity, targets, indication, trials, safety, sources)
            if indication.get("is_approved"):
                candidate.identity.is_fda_approved = True
            candidate.discovery_paths = list(entry.get("discovery_paths", []))
            return candidate

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_enrich_one, entry): entry["drug_name"] for entry in merged}
            for fut in as_completed(futures):
                try:
                    candidates.append(fut.result())
                except Exception as e:
                    logger.warning(f"Enrichment failed for {futures[fut]}: {e}")

        logger.info(f"Stage 2: Enriched {len(candidates)} candidates")
        print(f"        [Stage 2] Enriched {len(candidates)}/{len(merged)} candidates")

        # ── Post-enrichment brand→generic dedup (collect-then-apply) ──
        name_to_idx: Dict[str, int] = {}
        merge_ops: list = []
        for i, cand in enumerate(candidates):
            all_names = {cand.identity.drug_name.upper()}
            all_names.update(bn.upper() for bn in cand.identity.brand_names if bn)
            if cand.identity.chembl_id:
                all_names.add(cand.identity.chembl_id.upper())
            matched = False
            for name in all_names:
                if name in name_to_idx and name_to_idx[name] != i:
                    merge_ops.append((name_to_idx[name], i))
                    matched = True
                    break
            if not matched:
                for name in all_names:
                    name_to_idx.setdefault(name, i)
        remove_set: set = set()
        for keep_idx, remove_idx in merge_ops:
            if remove_idx in remove_set or keep_idx in remove_set:
                continue
            keep = candidates[keep_idx]
            dup = candidates[remove_idx]
            existing_genes = {t.gene_symbol.upper() for t in keep.targets}
            for t in dup.targets:
                if t.gene_symbol.upper() not in existing_genes:
                    keep.targets.append(t)
            keep.evidence_sources = list(set(keep.evidence_sources) | set(dup.evidence_sources))
            keep.discovery_paths = list(set(keep.discovery_paths) | set(dup.discovery_paths))
            for bn in dup.identity.brand_names:
                if bn and bn not in keep.identity.brand_names:
                    keep.identity.brand_names.append(bn)
            if not keep.identity.chembl_id:
                keep.identity.chembl_id = dup.identity.chembl_id
            # ── Propagate regulatory/clinical fields so SOC gate sees them ──
            if dup.identity.is_fda_approved:
                keep.identity.is_fda_approved = True
            if (dup.identity.max_phase or 0) > (keep.identity.max_phase or 0):
                keep.identity.max_phase = dup.identity.max_phase
            if dup.identity.indication_text:
                if keep.identity.indication_text:
                    if dup.identity.indication_text not in keep.identity.indication_text:
                        keep.identity.indication_text += "; " + dup.identity.indication_text
                else:
                    keep.identity.indication_text = dup.identity.indication_text
            if not keep.identity.pharm_class_epc and dup.identity.pharm_class_epc:
                keep.identity.pharm_class_epc = dup.identity.pharm_class_epc
            if not keep.identity.pharm_class_moa and dup.identity.pharm_class_moa:
                keep.identity.pharm_class_moa = dup.identity.pharm_class_moa
            if dup.trial_evidence and not keep.trial_evidence:
                keep.trial_evidence = dup.trial_evidence
            elif dup.trial_evidence and keep.trial_evidence:
                if (dup.trial_evidence.total_trials or 0) > (keep.trial_evidence.total_trials or 0):
                    keep.trial_evidence = dup.trial_evidence
            if dup.safety and not keep.safety:
                keep.safety = dup.safety
            remove_set.add(remove_idx)
            logger.warning(f"Post-enrichment dedup: merged '{dup.identity.drug_name}' into '{keep.identity.drug_name}'")
        if remove_set:
            candidates = [c for i, c in enumerate(candidates) if i not in remove_set]
            print(f"        [Stage 2 dedup] Merged {len(remove_set)} brand/generic duplicate(s)")

        # ── STAGE 2.5: Withdrawn Drug Filter ─────────────────────────────
        contraindicated = []
        active_candidates = []
        for candidate in candidates:
            if candidate.identity.withdrawn:
                reason = "Drug has been withdrawn from market"
                candidate.contraindication_flags.append(reason)
                candidate.contraindication_entries.append(ContraindicationEntry(
                    tier=1, reason=reason, source="withdrawn"))
                contraindicated.append(candidate)
            else:
                active_candidates.append(candidate)
        if contraindicated:
            logger.info(f"Stage 2.5: {len(contraindicated)} withdrawn drugs moved to contraindicated")
            print(f"        [Stage 2.5] {len(contraindicated)} withdrawn drugs filtered")
        candidates = active_candidates

        # ── STAGE 2.75: SOC Identification (dynamic) ─────────────────
        scoring_cfg = request.scoring_config or ScoringConfig()
        for candidate in candidates:
            if not (candidate.identity.is_fda_approved
                    and candidate.identity.max_phase is not None
                    and candidate.identity.max_phase >= 4):
                continue
            if scoring_cfg.use_soc_composite:
                conf = self._compute_soc_confidence(candidate, request, scoring_cfg)
                candidate.soc_confidence = conf
                candidate.is_soc_candidate = conf >= 0.50
            else:
                if self.scorer._has_disease_indication(candidate, request, strict=True):
                    candidate.is_soc_candidate = True
                    candidate.soc_confidence = 1.0
        soc_count = sum(1 for c in candidates if c.is_soc_candidate)
        if soc_count:
            print(f"        [Stage 2.75] {soc_count} SOC backbone candidates identified")

        # ── STAGE 3: Contraindication Check ──────────────────────────
        safe_candidates = []

        for candidate in candidates:
            # SOC drugs are shielded from contraindication — expression data informs but never overrides
            if candidate.is_soc_candidate:
                self._collect_soc_advisories(candidate, request)
                safe_candidates.append(candidate)
                continue
            hard_contra = False

            # Path 1: Gene-based — drug targets a downregulated gene in a potentially harmful way
            for gene in request.get_downregulated_genes_significant():
                target_genes = {t.gene_symbol.upper() for t in candidate.targets}
                if gene.gene_symbol.upper() in target_genes:
                    check = self.router.check_contraindication(
                        candidate.identity.drug_name, gene.gene_symbol, gene.direction, gene.log2fc)
                    if check.get("is_contraindicated"):
                        entry = ContraindicationEntry(
                            tier=check.get("tier", 2), reason=check["reason"],
                            source="gene_based", gene_symbol=gene.gene_symbol, log2fc=gene.log2fc)
                        candidate.contraindication_entries.append(entry)
                        if entry.tier <= 2:
                            candidate.contraindication_flags.append(entry.reason)
                            hard_contra = True
                        else:
                            candidate.caution_notes.append(entry)

            # Path 2: Biomarker-aware — drug requires a receptor the patient lacks (Tier 2)
            if request.biomarkers and not hard_contra:
                bio_flags = self._check_biomarker_contraindications(candidate, request.biomarkers)
                if bio_flags:
                    for reason in bio_flags:
                        entry = ContraindicationEntry(
                            tier=2, reason=reason, source="biomarker")
                        candidate.contraindication_entries.append(entry)
                    candidate.contraindication_flags.extend(bio_flags)
                    hard_contra = True
                    candidate.identity.genetic_eligibility_required = False

            # Path 3: Disease-AE — Tier 1 (drug causes/worsens disease)
            if not hard_contra and candidate.safety:
                alias_set = {a.lower() for a in ([request.disease] + (request.disease_aliases or []))}
                ind_text = (candidate.identity.indication_text or "").lower()
                treats_disease = ind_text and any(_disease_matches(a, ind_text) for a in alias_set)
                if not treats_disease:
                    ae_check = self.router.check_disease_in_adverse_events(
                        candidate.safety.__dict__, request.disease)
                    if ae_check.get("is_contraindicated"):
                        entry = ContraindicationEntry(
                            tier=1, reason=ae_check["reason"], source="disease_ae")
                        candidate.contraindication_entries.append(entry)
                        candidate.contraindication_flags.append(ae_check["reason"])
                        hard_contra = True

            # Path 4: Trial why_stopped — Tier 2
            if not hard_contra and candidate.trial_evidence:
                disease_tokens = {w.lower() for w in request.disease.split() if len(w) >= 4}
                for trial in candidate.trial_evidence.top_trials:
                    why = (trial.get("why_stopped") or "").lower()
                    if not why:
                        continue
                    if request.disease.lower() in why or (
                        disease_tokens and len(disease_tokens & set(why.split())) >= max(1, len(disease_tokens) // 2)
                    ):
                        reason = f"Trial stopped due to {request.disease}-related concerns: {trial.get('why_stopped', '')[:120]}"
                        entry = ContraindicationEntry(
                            tier=2, reason=reason, source="trial_stopped")
                        candidate.contraindication_entries.append(entry)
                        candidate.contraindication_flags.append(reason)
                        hard_contra = True
                        break

            if hard_contra:
                contraindicated.append(candidate)
            else:
                safe_candidates.append(candidate)

        # ── STAGE 4: Scoring ─────────────────────────────────────────────
        scoring_config = request.scoring_config or ScoringConfig()
        for candidate in safe_candidates:
            candidate.score = self.scorer.score(candidate, request)

        # Also score contraindicated ones for reference
        for candidate in contraindicated:
            candidate.score = self.scorer.score(candidate, request)

        # Tier-weighted contraindication multiplier (feature-flagged)
        if scoring_config.apply_contra_multipliers:
            for candidate in contraindicated:
                if candidate.score and candidate.contraindication_entries:
                    worst_tier = min(e.tier for e in candidate.contraindication_entries)
                    mult = scoring_config.contra_tier_multipliers.get(worst_tier, 0.25)
                    candidate.score.composite_score *= mult

        print(f"        [Stage 3-4] {len(safe_candidates)} safe + {len(contraindicated)} contraindicated | scoring complete")

        # ── STAGE 4.5: Claude Semantic Validation ─────────────────────────
        # Disease-discovered drugs scoring ≥ moderate are most prone to
        # semantic false positives (e.g., cardiac drug for skeletal disease).
        # Claude classifies relevance; mismatches are reclassified, not rescored.
        reclassified = self._validate_drug_relevance(safe_candidates, request)
        if reclassified:
            safe_candidates = [c for c in safe_candidates if not c.validation_caveat]
            print(f"        [Stage 4.5] Claude validation reclassified {len(reclassified)} drug(s)")

        # ── STAGE 5: Sort, Relevance Split, and Return ────────────────────
        # SOC drugs sort to front, then by composite score
        safe_candidates.sort(key=lambda c: (
            not c.is_soc_candidate,
            -(c.score.composite_score if c.score else 0),
        ))
        # Filter noise: require composite ≥ 10 and at least 1 positive component
        pre_filter_count = len(safe_candidates)
        safe_candidates = [c for c in safe_candidates if self._has_signal(c, scoring_config)]

        # Split: drugs with disease-treatment evidence vs gene-targeted only
        validated = [c for c in safe_candidates if c.score and c.score.disease_relevant]
        gene_only = [c for c in safe_candidates if c.score and not c.score.disease_relevant]
        gene_only.extend(reclassified)  # Claude-reclassified drugs go here with caveat
        final = validated[:request.max_results]

        print(f"        [Stage 5] {pre_filter_count} scored → {len(safe_candidates)} signal "
              f"→ {len(validated)} validated + {len(gene_only)} gene-targeted-only → {len(final)} returned")
        print(f"          Validated: {[(c.identity.drug_name, round(c.score.composite_score,1)) for c in validated]}")
        print(f"          Gene-only top10: {[(c.identity.drug_name, round(c.score.composite_score,1)) for c in gene_only[:10]]}")
        if final:
            top = final[0]
            print(f"          Top: {top.identity.drug_name} (score={top.score.composite_score:.0f}, "
                  f"dir={top.score.target_direction_match:.0f}, mag={top.score.target_magnitude_match:.0f}, "
                  f"clin={top.score.clinical_regulatory_score:.0f}, "
                  f"ot={top.score.ot_association_score:.0f}, pw={top.score.pathway_concordance:.0f})")

        return DrugQueryResponse(
            success=True, disease=request.disease,
            query_type=request.query_type.value,
            recommendations=final,
            contraindicated=contraindicated,
            gene_targeted_only=gene_only,
            metadata={
                "collections_queried": sorted(self.router.get_queried_collections()),
                "candidates_discovered": len(merged),
                "candidates_enriched": len(candidates),
                "candidates_contraindicated": len(contraindicated),
                "candidates_scored": len(safe_candidates),
            },
        )

    def _validate_drug(self, request: DrugQueryRequest) -> DrugQueryResponse:
        """Enrich and score a single drug against the molecular profile."""
        if not request.drug_name:
            return DrugQueryResponse(success=False, disease=request.disease,
                                     query_type=request.query_type.value,
                                     errors=["drug_name required for VALIDATE_DRUG"])

        identity = self.router.get_drug_identity(request.drug_name)
        targets = self.router.get_drug_targets(request.drug_name)
        for t in targets:
            if t.get("gene_symbol"):
                t["ot_association_score"] = self.router.get_target_disease_score(t["gene_symbol"], request.disease)

        indication = self.router.get_indication_status(request.drug_name, request.disease)
        identity["indication_text"] = indication.get("indication_text", "")
        trials = self.router.get_trial_evidence(request.drug_name, request.disease) if request.include_trials else {}
        safety = self.router.get_safety_profile(request.drug_name) if request.include_safety else {}

        candidate = self.aggregator.build_candidate(
            request.drug_name, identity, targets, indication, trials, safety)
        if indication.get("is_approved"):
            candidate.identity.is_fda_approved = True
        candidate.score = self.scorer.score(candidate, request)

        # SOC shield: standard-of-care drugs bypass contraindication gating
        scoring_cfg = request.scoring_config or ScoringConfig()
        if (candidate.identity.is_fda_approved
                and candidate.identity.max_phase is not None
                and candidate.identity.max_phase >= 4):
            if scoring_cfg.use_soc_composite:
                conf = self._compute_soc_confidence(candidate, request, scoring_cfg)
                candidate.soc_confidence = conf
                candidate.is_soc_candidate = conf >= 0.50
            elif self.scorer._has_disease_indication(candidate, request, strict=True):
                candidate.is_soc_candidate = True
                candidate.soc_confidence = 1.0
        if candidate.is_soc_candidate:
            self._collect_soc_advisories(candidate, request)
            return DrugQueryResponse(
                success=True, disease=request.disease,
                query_type=request.query_type.value,
                recommendations=[candidate])

        # Check contraindications
        for gene in request.get_downregulated_genes_significant():
            target_genes = {t.gene_symbol.upper() for t in candidate.targets}
            if gene.gene_symbol.upper() in target_genes:
                check = self.router.check_contraindication(
                    request.drug_name, gene.gene_symbol, gene.direction, gene.log2fc)
                if check.get("is_contraindicated"):
                    entry = ContraindicationEntry(
                        tier=check.get("tier", 2), reason=check["reason"],
                        source="gene_based", gene_symbol=gene.gene_symbol, log2fc=gene.log2fc)
                    candidate.contraindication_entries.append(entry)
                    if entry.tier <= 2:
                        candidate.contraindication_flags.append(check["reason"])
                    else:
                        candidate.caution_notes.append(entry)

        # Biomarker contraindications
        if request.biomarkers:
            bio_flags = self._check_biomarker_contraindications(candidate, request.biomarkers)
            for reason in bio_flags:
                candidate.contraindication_entries.append(ContraindicationEntry(
                    tier=2, reason=reason, source="biomarker"))
            candidate.contraindication_flags.extend(bio_flags)

        worst_tier = min((e.tier for e in candidate.contraindication_entries), default=None)
        resp = DrugQueryResponse(success=True, disease=request.disease, query_type=request.query_type.value)
        if worst_tier is not None and worst_tier <= 2:
            resp.contraindicated = [candidate]
        else:
            resp.recommendations = [candidate]
        return resp

    def _check_contraindication(self, request: DrugQueryRequest) -> DrugQueryResponse:
        """Quick contraindication check for a drug against patient genes."""
        if not request.drug_name:
            return DrugQueryResponse(success=False, disease=request.disease,
                                     query_type=request.query_type.value,
                                     errors=["drug_name required"])

        targets = self.router.get_drug_targets(request.drug_name)
        candidate = DrugCandidate(identity=DrugIdentity(drug_name=request.drug_name))
        for t in targets:
            candidate.targets.append(TargetEvidence(
                gene_symbol=t.get("gene_symbol", ""), action_type=t.get("action_type", "UNKNOWN")))

        for gene in request.genes:
            check = self.router.check_contraindication(
                request.drug_name, gene.gene_symbol, gene.direction, gene.log2fc)
            if check.get("is_contraindicated"):
                entry = ContraindicationEntry(
                    tier=check.get("tier", 2), reason=check["reason"],
                    source="gene_based", gene_symbol=gene.gene_symbol, log2fc=gene.log2fc)
                candidate.contraindication_entries.append(entry)
                candidate.contraindication_flags.append(check["reason"])

        resp = DrugQueryResponse(success=True, disease=request.disease, query_type=request.query_type.value)
        worst_tier = min((e.tier for e in candidate.contraindication_entries), default=None)
        if worst_tier is not None and worst_tier <= 2:
            resp.contraindicated = [candidate]
        else:
            resp.recommendations = [candidate]
        return resp

    def _safety_profile(self, request: DrugQueryRequest) -> DrugQueryResponse:
        """Full safety data for a drug."""
        if not request.drug_name:
            return DrugQueryResponse(success=False, disease=request.disease,
                                     query_type=request.query_type.value,
                                     errors=["drug_name required"])

        safety = self.router.get_safety_profile(request.drug_name)
        candidate = self.aggregator.build_candidate(
            request.drug_name, {"drug_name": request.drug_name}, [], {}, {}, safety)

        return DrugQueryResponse(
            success=True, disease=request.disease,
            query_type=request.query_type.value,
            recommendations=[candidate],
        )

    def _drug_details(self, request: DrugQueryRequest) -> DrugQueryResponse:
        """Everything from all 15 collections about a drug."""
        if not request.drug_name:
            return DrugQueryResponse(success=False, disease=request.disease,
                                     query_type=request.query_type.value,
                                     errors=["drug_name required"])

        identity = self.router.get_drug_identity(request.drug_name)
        targets = self.router.get_drug_targets(request.drug_name)
        for t in targets:
            if t.get("gene_symbol"):
                t["ot_association_score"] = self.router.get_target_disease_score(t["gene_symbol"], request.disease)
        indication = self.router.get_indication_status(request.drug_name, request.disease)
        identity["indication_text"] = indication.get("indication_text", "")
        trials = self.router.get_trial_evidence(request.drug_name, request.disease)
        safety = self.router.get_safety_profile(request.drug_name)

        candidate = self.aggregator.build_candidate(
            request.drug_name, identity, targets, indication, trials, safety)
        if indication.get("is_approved"):
            candidate.identity.is_fda_approved = True
        candidate.score = self.scorer.score(candidate, request)

        return DrugQueryResponse(
            success=True, disease=request.disease,
            query_type=request.query_type.value,
            recommendations=[candidate],
        )

    # ── Synonym Resolution ─────────────────────────────────────────────

    @staticmethod
    def _build_synonym_list(drug_name: str, identity: Dict,
                            original_names: List[str]) -> List[str]:
        """Build a concise list of alternative names for fallback queries.

        Includes ChEMBL canonical name (INN), original discovery names, and
        brand names.  Limited to top 3 by shortest length to prefer INNs and
        concise identifiers over verbose salt-form names.
        """
        drug_upper = drug_name.upper()
        candidates: Dict[str, str] = {}  # upper → original-case

        # 1. ChEMBL canonical drug_name (often the INN, e.g. ASCIMINIB)
        chembl_dn = identity.get("chembl_drug_name", "")
        if chembl_dn and chembl_dn.upper() != drug_upper:
            candidates[chembl_dn.upper()] = chembl_dn

        # 2. Original discovery names from merge
        for on in original_names:
            ou = on.upper().strip()
            if ou and ou != drug_upper and ou not in candidates:
                candidates[ou] = on

        # 3. Brand names / synonyms from identity
        for bn in identity.get("brand_names", []):
            bu = bn.upper().strip()
            if bu and bu != drug_upper and bu not in candidates:
                candidates[bu] = bn

        # Return top 3 by shortest name length (prefer INN over long codes)
        sorted_syns = sorted(candidates.values(), key=len)
        return sorted_syns[:3]

    # ── Biomarker Contraindication ─────────────────────────────────────

    # Receptor/biomarker keywords that map to indication text patterns
    _BIOMARKER_INDICATION_KEYWORDS = {
        "ER":  ["estrogen receptor", "hormone receptor", "er-positive", "er+"],
        "PR":  ["progesterone receptor", "hormone receptor", "pr-positive", "pr+"],
        "HER2": ["her2", "erbb2", "her2-overexpressing", "her2-positive", "her2+"],
        "AR":  ["androgen receptor", "ar-positive"],
    }

    def _compute_soc_confidence(self, candidate: DrugCandidate,
                                request: DrugQueryRequest,
                                config: ScoringConfig) -> float:
        """Multi-signal SOC confidence: indication_sim + pharm_class_sim + clinical_depth."""
        weights = config.soc_signal_weights

        # Signal 1: indication similarity (0.0–1.0)
        ind_sim = self.scorer._indication_similarity(candidate, request)

        # Signal 2: pharmacological class semantic similarity (0.0–1.0)
        pharm_sim = 0.0
        epc = candidate.identity.pharm_class_epc or ""
        if epc and self.scorer.embedder:
            try:
                epc_vec = self.scorer.embedder.encode(epc)
                dis_vec = self.scorer.embedder.encode(request.disease)
                pharm_sim = max(0.0, float(np.dot(
                    epc_vec / (np.linalg.norm(epc_vec) + 1e-10),
                    dis_vec / (np.linalg.norm(dis_vec) + 1e-10),
                )))
            except Exception:
                pass

        # Signal 3: clinical evidence depth (0.0–1.0)
        clinical_depth = 0.0
        te = candidate.trial_evidence
        if te and te.total_trials > 0:
            trial_score = min(1.0, te.total_trials / 10.0) * 0.4
            completed_frac = (te.completed_trials / te.total_trials) * 0.3 if te.total_trials > 0 else 0.0
            results_frac = (te.trials_with_results / te.total_trials) * 0.3 if te.total_trials > 0 else 0.0
            clinical_depth = trial_score + completed_frac + results_frac

        composite = (
            weights.get("indication_sim", 0.40) * ind_sim
            + weights.get("pharm_class_sim", 0.25) * pharm_sim
            + weights.get("clinical_depth", 0.35) * clinical_depth
        )
        return round(composite, 3)

    def _collect_soc_advisories(self, candidate: DrugCandidate, request: DrugQueryRequest):
        """Run all contraindication paths for SOC drugs, recording as advisories, not blocks."""
        drug = candidate.identity.drug_name
        disease = request.disease

        # Path 1: Gene-based
        for gene in request.get_downregulated_genes_significant():
            target_genes = {t.gene_symbol.upper() for t in candidate.targets}
            if gene.gene_symbol.upper() in target_genes:
                check = self.router.check_contraindication(drug, gene.gene_symbol, gene.direction, gene.log2fc)
                if check.get("is_contraindicated"):
                    candidate.soc_advisory_notes.append(
                        f"Target {gene.gene_symbol} is downregulated (log2FC: {gene.log2fc:.2f}) — "
                        f"noted as advisory; drug retained as backbone therapy for {disease}")

        # Path 2: Biomarker
        if request.biomarkers:
            bio_flags = self._check_biomarker_contraindications(candidate, request.biomarkers)
            for reason in bio_flags:
                candidate.soc_advisory_notes.append(
                    f"Biomarker advisory: {reason} — retained as SOC for {disease}; "
                    f"confirmatory testing recommended")

        # Path 3: Disease-AE (FAERS)
        if candidate.safety:
            ae_check = self.router.check_disease_in_adverse_events(candidate.safety.__dict__, disease)
            if ae_check.get("is_contraindicated"):
                candidate.soc_advisory_notes.append(
                    f"FAERS contains {disease}-related AE reports for {drug}. "
                    f"This reflects the treated population, not a risk of {drug} itself")

        # Path 4: Trial stopped
        if candidate.trial_evidence:
            disease_tokens = {w.lower() for w in disease.split() if len(w) >= 4}
            for trial in candidate.trial_evidence.top_trials:
                why = (trial.get("why_stopped") or "").lower()
                if not why:
                    continue
                if disease.lower() in why or (
                    disease_tokens and len(disease_tokens & set(why.split())) >= max(1, len(disease_tokens) // 2)
                ):
                    candidate.soc_advisory_notes.append(
                        f"Trial note: {trial.get('why_stopped', '')[:120]} — "
                        f"retained as SOC for {disease}; clinical monitoring advised")
                    break

    def _check_biomarker_contraindications(
        self, candidate: DrugCandidate, biomarkers: list
    ) -> List[str]:
        """Flag drugs whose indication requires a receptor the patient lacks."""
        flags = []
        # Gather drug's therapeutic context
        texts = []
        if candidate.identity.indication_text:
            texts.append(candidate.identity.indication_text)
        if candidate.identity.pharm_class_epc:
            texts.append(candidate.identity.pharm_class_epc)
        if candidate.identity.pharm_class_moa:
            texts.append(candidate.identity.pharm_class_moa)
        for t in candidate.targets:
            if t.fda_moa_narrative:
                texts.append(t.fda_moa_narrative[:300])
        drug_context = " ".join(texts).lower().replace("-", " ")
        if not drug_context.strip():
            return flags

        for bio in biomarkers:
            if bio.status != "negative":
                continue  # Only flag when patient LACKS a required receptor
            if getattr(bio, 'biomarker_type', None) == 'B':
                continue  # Type B: RNA cannot determine receptor status
            bio_name = bio.biomarker_name.upper()
            keywords = self._BIOMARKER_INDICATION_KEYWORDS.get(bio_name, [bio.biomarker_name.lower()])

            if any(kw in drug_context for kw in keywords):
                # Drug's indication mentions this receptor → patient lacks it
                # Extract the most relevant snippet for traceability
                snippet = ""
                for txt in texts:
                    txt_lower = txt.lower()
                    for kw in keywords:
                        idx = txt_lower.find(kw)
                        if idx >= 0:
                            start = max(0, idx - 30)
                            end = min(len(txt), idx + len(kw) + 50)
                            snippet = txt[start:end].strip()
                            break
                    if snippet:
                        break

                reason = (
                    f"Contraindicated: {candidate.identity.drug_name} requires "
                    f"{bio_name}-positive status but patient is {bio_name}-negative. "
                    f"Source: \"{snippet}\""
                )
                flags.append(reason)
                logger.info(f"Biomarker contra: {candidate.identity.drug_name} vs {bio_name}-negative")

        return flags

    # ── Claude Semantic Validation ───────────────────────────────────────

    _validation_cache: Dict = {}

    def _validate_drug_relevance(
        self, candidates: List[DrugCandidate], request: DrugQueryRequest
    ) -> List[DrugCandidate]:
        """Validate disease-discovered drugs via Claude to catch semantic mismatches.

        Returns list of reclassified candidates (those Claude deemed irrelevant).
        Score is preserved; only validation_caveat is set and the drug is moved
        to gene_targeted_only by the caller.
        """
        from .schemas import DRUG_MODERATE_PRIORITY_THRESHOLD

        to_validate = [
            c for c in candidates
            if c.score and c.score.composite_score >= DRUG_MODERATE_PRIORITY_THRESHOLD
            and "disease" in c.discovery_paths
        ]
        if not to_validate:
            return []

        # Build LLM client (fail-open: if unavailable, skip validation entirely)
        try:
            from agentic_ai_wf.reporting_pipeline_agent.llm_factory import create_llm_client
            llm = create_llm_client()
            if not llm:
                return []
        except Exception as e:
            logger.warning(f"Claude validation skipped — LLM unavailable: {e}")
            return []

        # Check cache and build batch for uncached drugs
        disease_key = request.disease.lower()
        uncached = []
        for c in to_validate:
            cache_key = (c.identity.drug_name.lower(), disease_key)
            if cache_key in self._validation_cache:
                result = self._validation_cache[cache_key]
                if not result.get("relevant", True):
                    c.validation_caveat = result.get("reason", "")
            else:
                uncached.append(c)

        if uncached:
            drugs_info = []
            for c in uncached:
                moa = c.identity.pharm_class_moa or next(
                    (t.mechanism_of_action for t in c.targets if t.mechanism_of_action), "unknown")
                ind = (c.identity.indication_text or "")[:200]
                drugs_info.append(f"- {c.identity.drug_name}: mechanism={moa}, indication={ind}")

            prompt = (
                f"You are a clinical pharmacology expert. For each drug below, assess whether "
                f"it is semantically appropriate for treating **{request.disease}** specifically — "
                f"not a related but distinct condition or organ system.\n\n"
                f"Also check for CLASS-EFFECT CONTRAINDICATIONS: if a drug belongs to a "
                f"pharmacological class that is known to be contraindicated, harmful, or "
                f"ineffective in {request.disease} (e.g., TNF inhibitors in SLE, "
                f"immunosuppressants in active infections), flag it as not relevant and "
                f"explain the class-effect concern.\n\n"
                f"DRUGS:\n" + "\n".join(drugs_info) + "\n\n"
                f"Return ONLY a JSON object:\n"
                f'{{"DrugName": {{"relevant": true/false, "reason": "brief explanation", '
                f'"class_effect_concern": "optional: class-level contraindication if any"}}, ...}}'
            )

            try:
                import json as _json
                response = llm.chat.completions.create(
                    model=getattr(llm, 'model_name', None),
                    messages=[
                        {"role": "system", "content": "You are a clinical pharmacology expert. Return only JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                    task_type="critical",
                )
                content = response.choices[0].message.content.strip()
                if '```' in content:
                    content = content.split('```')[1].lstrip('json\n')
                    if '```' in content:
                        content = content.split('```')[0]
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    verdicts = _json.loads(json_match.group())
                    for c in uncached:
                        v = verdicts.get(c.identity.drug_name, {})
                        cache_key = (c.identity.drug_name.lower(), disease_key)
                        self._validation_cache[cache_key] = v
                        if not v.get("relevant", True):
                            caveat = v.get("reason", "Semantic mismatch detected")
                            class_concern = v.get("class_effect_concern", "")
                            if class_concern:
                                caveat = f"{caveat} [Class-effect: {class_concern}]"
                            c.validation_caveat = caveat
            except Exception as e:
                logger.warning(f"Claude drug validation failed (fail-open): {e}")

        reclassified = [c for c in to_validate if c.validation_caveat]
        if reclassified:
            logger.info(f"Claude validation: {len(reclassified)}/{len(to_validate)} drugs reclassified "
                        f"for {request.disease}")
        return reclassified

    # ── Convenience Methods ──────────────────────────────────────────────

    def get_safety_profile(self, drug_name: str) -> SafetyProfile:
        raw = self.router.get_safety_profile(drug_name)
        return SafetyProfile(**{k: v for k, v in raw.items() if k in SafetyProfile.__dataclass_fields__})

    def find_drugs_for_gene(self, gene_symbol: str, top_k: int = 10) -> List[Dict]:
        return self.router.find_drugs_for_target(gene_symbol, top_k)

    def find_drugs_for_disease(self, disease_name: str, top_k: int = 10) -> List[Dict]:
        return self.router.find_drugs_for_disease(disease_name, top_k)

    def get_disease_aliases(self, disease_name: str) -> List[str]:
        return self.router.get_disease_aliases(disease_name)

    def get_trial_evidence(self, drug_name: str, disease_name: str) -> TrialEvidence:
        raw = self.router.get_trial_evidence(drug_name, disease_name)
        return TrialEvidence(**{k: v for k, v in raw.items() if k in TrialEvidence.__dataclass_fields__})

    def get_drug_targets(self, drug_name: str) -> List[TargetEvidence]:
        raw = self.router.get_drug_targets(drug_name)
        return [TargetEvidence(
            gene_symbol=t.get("gene_symbol", ""),
            action_type=t.get("action_type", "UNKNOWN"),
            mechanism_of_action=t.get("mechanism"),
            fda_moa_narrative=t.get("fda_narrative"),
        ) for t in raw]

    def get_drug_identity(self, drug_name: str) -> DrugIdentity:
        raw = self.router.get_drug_identity(drug_name)
        return DrugIdentity(**{k: v for k, v in raw.items() if k in DrugIdentity.__dataclass_fields__})

    def get_pathway_drugs(self, pathway_name: str, top_k: int = 10) -> List[Dict]:
        return self.router.get_pathway_drugs(pathway_name, top_k=top_k)

    # ── Operational ──────────────────────────────────────────────────────

    def health_check(self) -> Dict:
        from .collection_router import ALL_COLLECTIONS
        available = sorted(self.router._available)
        unavailable = sorted(set(ALL_COLLECTIONS) - self.router._available)
        total = sum(self.router._collection_counts.get(c, 0) for c in available)

        return {
            "status": "healthy" if len(available) >= 10 else "degraded",
            "available_collections": available,
            "unavailable_collections": unavailable,
            "total_points": total,
            "per_collection": {c: self.router._collection_counts.get(c, 0) for c in available},
            "ensembl_cache_loaded": len(self.router._ensembl_cache) > 0,
            "ensembl_cache_size": len(self.router._ensembl_cache),
            "embedder_device": str(self.router.embedder.device),
        }

    def get_capabilities(self) -> Dict:
        return {
            "supported_query_types": [qt.value for qt in QueryType],
            "available_collections": sorted(self.router._available),
            "convenience_methods": [
                "get_safety_profile", "find_drugs_for_gene", "find_drugs_for_disease",
                "get_trial_evidence", "get_drug_targets", "get_drug_identity", "get_pathway_drugs",
            ],
            "scoring_weights": {
                "target_direction_weight": self.scorer.config.target_direction_weight,
                "target_magnitude_weight": self.scorer.config.target_magnitude_weight,
                "clinical_regulatory_weight": self.scorer.config.clinical_regulatory_weight,
                "ot_weight": self.scorer.config.ot_weight,
                "pathway_weight": self.scorer.config.pathway_weight,
                "safety_max_penalty": self.scorer.config.safety_max_penalty,
            },
        }

    def shutdown(self):
        self.router.shutdown()

    # TODO: LLM Integration — add _generate_narrative(candidate, request) method
    # using BedrockLLMClient from reporting_pipeline_agent.llm_factory.create_llm_client()
    # to produce human-readable recommendation narratives per DrugCandidate.
    # Wire into Stage 5 as optional post-scoring step (gated by request flag).


# ── Singleton Factory ────────────────────────────────────────────────────────

_service_instance: Optional[DrugAgentService] = None


def get_service(scoring_config: Optional[ScoringConfig] = None) -> DrugAgentService:
    global _service_instance
    if _service_instance is None:
        _service_instance = DrugAgentService(scoring_config=scoring_config)
    return _service_instance
