"""Routes queries to all 15 Qdrant collections with parallel execution."""

import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

logger = logging.getLogger(__name__)


def _disease_matches(alias: str, text: str) -> bool:
    """Word-boundary match to prevent 'myopathy' matching 'cardiomyopathy'."""
    if not alias or not text:
        return False
    return bool(re.search(r'\b' + re.escape(alias) + r'\b', text, re.IGNORECASE))


ALL_COLLECTIONS = [
    "Drug_agent", "ChEMBL_drugs", "Raw_csv_KG",
    "OpenTargets_data", "OpenTargets_drugs_enriched",
    "OpenTargets_adverse_events", "OpenTargets_pharmacogenomics",
    "FDA_Orange_Book", "FDA_DrugsFDA", "FDA_FAERS",
    "FDA_Drug_Labels", "FDA_Enforcement",
    "ClinicalTrials_summaries", "ClinicalTrials_results",
]


class CollectionRouter:

    def __init__(self):
        from ..opentargets.ot_base import get_qdrant, get_embedder
        self.client = get_qdrant()
        self.embedder = get_embedder()
        self._embed_cache: Dict[str, list] = {}
        self._ensembl_cache: Dict[str, str] = {}
        self._available: set = set()
        self._collection_counts: Dict[str, int] = {}
        self._queried_collections: set = set()
        self._executor = ThreadPoolExecutor(max_workers=8)

        self._init_available_collections()
        self._load_ensembl_cache()
        self._ot_score_fields: list = []
        self._probe_ot_schema()
        logger.info(f"CollectionRouter ready — {len(self._available)}/{len(ALL_COLLECTIONS)} collections available")

    # ── Infrastructure ───────────────────────────────────────────────────────

    def _init_available_collections(self):
        try:
            for c in self.client.get_collections().collections:
                if c.name in ALL_COLLECTIONS:
                    self._available.add(c.name)
                    try:
                        self._collection_counts[c.name] = self.client.count(c.name).count
                    except Exception:
                        self._collection_counts[c.name] = -1
            missing = set(ALL_COLLECTIONS) - self._available
            if missing:
                logger.warning(f"Unavailable collections: {missing}")
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")

    def _load_ensembl_cache(self):
        if "OpenTargets_data" not in self._available:
            logger.warning("OpenTargets_data unavailable — Ensembl cache empty")
            return
        try:
            offset = None
            filt = Filter(must=[FieldCondition(key="entity_type", match=MatchValue(value="target"))])
            while True:
                points, offset = self.client.scroll(
                    "OpenTargets_data", scroll_filter=filt, limit=1000,
                    with_payload=["id", "name"], offset=offset,
                )
                for p in points:
                    gene = p.payload.get("name", "")
                    ens = p.payload.get("id", "")
                    if gene and ens:
                        self._ensembl_cache[gene.upper()] = ens
                if offset is None:
                    break
            logger.info(f"Ensembl cache loaded: {len(self._ensembl_cache):,} genes")
        except Exception as e:
            logger.warning(f"Ensembl cache load failed: {e}")

    def _probe_ot_schema(self):
        """Sample OpenTargets_data association records to discover available score fields."""
        if "OpenTargets_data" not in self._available:
            return
        try:
            filt = Filter(must=[FieldCondition(key="entity_type", match=MatchValue(value="association"))])
            points, _ = self.client.scroll("OpenTargets_data", scroll_filter=filt, limit=3, with_payload=True)
            all_keys: set = set()
            for p in points:
                all_keys.update(p.payload.keys())
            score_candidates = [k for k in sorted(all_keys) if "score" in k.lower() or "overall" in k.lower()]
            self._ot_score_fields = score_candidates
            logger.info(f"OT probe: {len(points)} samples, score-like fields={score_candidates}, all keys={sorted(all_keys)}")
        except Exception as e:
            logger.warning(f"OT schema probe failed: {e}")

    def _embed(self, text: str) -> list:
        if text in self._embed_cache:
            return self._embed_cache[text]
        vec = self.embedder.encode(text).tolist()
        self._embed_cache[text] = vec
        return vec

    @staticmethod
    def _ascii_fold(text: str) -> str:
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii').replace("'", "").replace("\u2019", "")

    def _search(self, collection: str, query_text: str,
                top_k: int = 10, qfilter: Optional[Filter] = None) -> List[Dict]:
        if collection not in self._available:
            return []
        self._queried_collections.add(collection)
        try:
            vec = self._embed(query_text)
            results = self.client.query_points(
                collection_name=collection, query=vec,
                query_filter=qfilter, limit=top_k, with_payload=True,
            )
            return [{"score": r.score, "payload": r.payload} for r in results.points]
        except Exception as e:
            logger.warning(f"Search failed on {collection}: {e}")
            return []

    def reset_query_tracking(self):
        self._queried_collections.clear()

    def get_queried_collections(self) -> set:
        return set(self._queried_collections)

    def _parallel_search(self, queries: List[Tuple[str, str, int, Optional[Filter]]]) -> Dict[str, List[Dict]]:
        """Execute multiple (collection, query_text, top_k, filter) searches in parallel."""
        results: Dict[str, List[Dict]] = {}
        futures = {}
        for coll, text, k, filt in queries:
            key = f"{coll}|{text[:40]}"
            futures[self._executor.submit(self._search, coll, text, k, filt)] = key
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                logger.warning(f"Parallel search failed for {key}: {e}")
                results[key] = []
        counts = {k.split('|')[0]: len(v) for k, v in results.items()}
        logger.debug(f"Parallel search: {counts}")
        return results

    # ── Gene-Gene Functional Relationships (KG) ─────────────────────────────

    _KG_GENE_TYPE = "gene/protein"

    def get_functionally_related_genes(self, gene_symbol: str) -> List[str]:
        """Resolve functionally related genes via Raw_csv_KG PPI triples + semantic search."""
        cache_key = f"_effectors_{gene_symbol.upper()}"
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        if "Raw_csv_KG" not in self._available:
            self._embed_cache[cache_key] = []
            return []

        related: set = set()
        gene_upper = gene_symbol.upper()

        # Payload-filtered scroll: exact gene name on either side of KG triple
        for side, other in [("x", "y"), ("y", "x")]:
            try:
                filt = Filter(must=[
                    FieldCondition(key=f"{side}_name", match=MatchValue(value=gene_symbol)),
                    FieldCondition(key=f"{side}_type", match=MatchValue(value=self._KG_GENE_TYPE)),
                    FieldCondition(key=f"{other}_type", match=MatchValue(value=self._KG_GENE_TYPE)),
                ])
                self._queried_collections.add("Raw_csv_KG")
                pts, _ = self.client.scroll(
                    "Raw_csv_KG", scroll_filter=filt, limit=200, with_payload=True)
                for p in pts:
                    name = (p.payload.get(f"{other}_name") or "").upper()
                    if name and name != gene_upper:
                        related.add(name)
            except Exception:
                pass

        # Semantic search for broader coverage (still filtered to gene↔gene triples)
        try:
            vec = self._embed(f"{gene_symbol} protein interaction signaling")
            self._queried_collections.add("Raw_csv_KG")
            results = self.client.query_points(
                "Raw_csv_KG", query=vec, limit=50, with_payload=True,
                query_filter=Filter(must=[
                    FieldCondition(key="x_type", match=MatchValue(value=self._KG_GENE_TYPE)),
                    FieldCondition(key="y_type", match=MatchValue(value=self._KG_GENE_TYPE)),
                ]),
            )
            for r in results.points:
                pl = r.payload
                for s in ("x", "y"):
                    name = (pl.get(f"{s}_name") or "").upper()
                    if name and name != gene_upper:
                        related.add(name)
        except Exception:
            pass

        result = sorted(related)
        self._embed_cache[cache_key] = result
        logger.debug(f"KG effectors for {gene_symbol}: {len(result)} related genes")
        return result

    # ── Stage 1: Candidate Discovery ─────────────────────────────────────────

    def find_drugs_for_target(self, gene_symbol: str, top_k: int = 10) -> List[Dict]:
        """Find drugs targeting a gene — queries 5 collections in parallel."""
        gene_up = gene_symbol.upper()
        semantic_q = f"{gene_symbol} inhibitor drug target"

        # ChEMBL uses a wider Qdrant limit because biosimilars and salt
        # forms often dominate the top-k, pushing genuine drugs below the
        # cut-off.  Other collections are less duplicative.
        chembl_limit = top_k * 3

        queries = [
            ("Drug_agent", f"{gene_symbol} drug target therapy",
             top_k, Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="gene_drug"))])),
            ("ChEMBL_drugs", semantic_q, chembl_limit, None),
            ("OpenTargets_drugs_enriched", f"{gene_symbol} drug mechanism", top_k, None),
            ("FDA_Drug_Labels", gene_symbol, top_k,
             Filter(must=[FieldCondition(key="section_name", match=MatchValue(value="mechanism_of_action"))])),
            ("Raw_csv_KG", f"{gene_symbol} drug target interaction", top_k, None),
        ]

        raw = self._parallel_search(queries)
        candidates = []
        seen = set()
        # Track chembl_ids for names first seen from non-ChEMBL sources,
        # so the ID isn't lost when ChEMBL re-discovers the same name.
        seen_chembl: Dict[str, str] = {}

        for key, hits in raw.items():
            coll = key.split("|")[0]
            for h in hits:
                p = h["payload"]
                drugs = self._extract_drug_names(p, coll)
                for dname in drugs:
                    norm = dname.upper().strip()
                    if not norm:
                        continue
                    if norm in seen:
                        if coll == "ChEMBL_drugs" and p.get("chembl_id"):
                            seen_chembl.setdefault(norm, p["chembl_id"])
                        continue

                    # Verify gene relevance for collections that have structured target data
                    if coll == "ChEMBL_drugs":
                        targets = [t.upper() for t in p.get("target_gene_symbols", [])]
                        if targets and gene_up not in targets:
                            continue
                    elif coll == "OpenTargets_drugs_enriched":
                        lt = [t.upper() for t in p.get("linked_targets", []) + p.get("mechanism_targets", [])]
                        ensembl = self._ensembl_cache.get(gene_up, "")
                        if lt and gene_up not in lt and ensembl not in [x.upper() for x in p.get("linked_targets", [])]:
                            continue
                    elif coll == "Raw_csv_KG":
                        if not self._kg_involves_entity(p, gene_up, {"gene", "protein"}):
                            continue

                    seen.add(norm)
                    candidates.append({
                        "drug_name": dname,
                        "source": coll,
                        "action_type": self._extract_action_type(p, coll),
                        "mechanism": p.get("mechanism_of_action", p.get("text_content", ""))[:200],
                        "phase": p.get("max_phase", p.get("max_phase", None)),
                        "score": h["score"],
                        "gene_symbol": gene_symbol,
                        "chembl_id": p.get("chembl_id") if coll == "ChEMBL_drugs" else None,
                    })

        # Backfill chembl_ids captured from seen-but-skipped ChEMBL hits
        for c in candidates:
            if not c.get("chembl_id"):
                cid = seen_chembl.get(c["drug_name"].upper().strip())
                if cid:
                    c["chembl_id"] = cid

        per_coll = {}
        for c in candidates:
            per_coll[c["source"]] = per_coll.get(c["source"], 0) + 1
        logger.info(f"find_drugs_for_target({gene_symbol}): {per_coll}")
        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k * 5]

    def find_drugs_for_disease(self, disease_name: str, top_k: int = 10,
                                disease_aliases: Optional[List[str]] = None) -> List[Dict]:
        """Find drugs associated with a disease — queries 5 collections per alias."""
        search_terms = [disease_name] + (disease_aliases or [])[:5]
        all_queries = []
        primary_keys: set = set()
        for i, term in enumerate(dict.fromkeys(t.lower() for t in search_terms)):
            # OT gets a larger top_k to catch drugs with secondary disease
            # indications that have lower vector similarity (e.g., Belimumab
            # stored under SLE but indicated for Sjögren's).
            ot_k = max(top_k * 3, 25)
            term_queries = [
                ("Drug_agent", f"{term} drug treatment therapy",
                 top_k, Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="disease_drug"))])),
                ("ClinicalTrials_summaries", f"{term} drug treatment", top_k, None),
                ("OpenTargets_drugs_enriched", term, ot_k, None),
                ("FDA_Drug_Labels", term, top_k,
                 Filter(must=[FieldCondition(key="section_name", match=MatchValue(value="indications_and_usage"))])),
                ("Raw_csv_KG", f"{term} treatment drug therapy", top_k, None),
            ]
            all_queries.extend(term_queries)
            if i == 0:
                primary_keys = {f"{c}|{t[:40]}" for c, t, _, _ in term_queries}

        raw = self._parallel_search(all_queries)
        drug_map: Dict[str, Dict] = {}
        # Build alias set for OT indication matching
        _alias_set = {self._ascii_fold(t).lower() for t in search_terms}

        for key, hits in raw.items():
            coll = key.split("|")[0]
            is_primary = key in primary_keys
            for h in hits:
                p = h["payload"]
                drugs = self._extract_drug_names(p, coll)

                # OT indication check: if this OT drug has the disease in its
                # indications list, force it as primary (disease-indicated drug)
                ot_indicated = False
                if coll == "OpenTargets_drugs_enriched":
                    for ind in p.get("indications", []):
                        ind_name = self._ascii_fold(ind.get("disease_name", "")).lower()
                        if any(_disease_matches(a, ind_name) or _disease_matches(ind_name, a) for a in _alias_set):
                            ot_indicated = True
                            break

                for dname in drugs:
                    norm = dname.upper().strip()
                    if not norm:
                        continue
                    if coll == "Raw_csv_KG" and not self._kg_involves_entity(p, "", {"drug", "compound"}):
                        continue
                    if norm in drug_map:
                        drug_map[norm]["score"] = max(drug_map[norm]["score"], h["score"])
                        if is_primary or ot_indicated:
                            drug_map[norm]["_primary"] = True
                        continue
                    drug_map[norm] = {
                        "drug_name": dname,
                        "source": coll,
                        "indication": disease_name,
                        "phase": p.get("max_phase"),
                        "score": h["score"],
                        "chembl_id": p.get("chembl_id") if coll == "ChEMBL_drugs" else None,
                        "_primary": is_primary or ot_indicated,
                    }

        # ── OT indication reverse scan ──
        # Vector search misses drugs whose OT entries are stored under
        # different primary diseases (e.g., Belimumab under SLE but
        # indicated for Sjögren's).  Use multiple disease-centric query
        # variations with high top_k to maximize OT coverage, then
        # filter to drugs with explicit indication match.
        ot_scan_queries = []
        for term in list(dict.fromkeys(t.lower() for t in search_terms))[:3]:
            ot_scan_queries.extend([
                ("OpenTargets_drugs_enriched", f"{term} treatment drug therapy indication", 50, None),
                ("OpenTargets_drugs_enriched", f"{term} approved drug FDA", 50, None),
                ("OpenTargets_drugs_enriched", f"autoimmune {term} monoclonal antibody biologic", 30, None),
            ])
        ot_raw = self._parallel_search(ot_scan_queries)
        ot_indicated_count = 0
        for key, hits in ot_raw.items():
            for h in hits:
                p = h["payload"]
                for ind in p.get("indications", []):
                    ind_name = self._ascii_fold(ind.get("disease_name", "")).lower()
                    if not any(_disease_matches(a, ind_name) or _disease_matches(ind_name, a) for a in _alias_set):
                        continue
                    drugs = self._extract_drug_names(p, "OpenTargets_drugs_enriched")
                    for dname in drugs:
                        norm = dname.upper().strip()
                        if not norm:
                            continue
                        if norm not in drug_map:
                            drug_map[norm] = {
                                "drug_name": dname,
                                "source": "OpenTargets_drugs_enriched",
                                "indication": disease_name,
                                "phase": ind.get("phase") or p.get("max_phase"),
                                "score": h["score"],
                                "chembl_id": p.get("chembl_id"),
                                "_primary": True,
                            }
                            ot_indicated_count += 1
                        else:
                            drug_map[norm]["_primary"] = True
                    break
        if ot_indicated_count:
            logger.info(f"OT indication scan: found {ot_indicated_count} new indicated drugs")

        # ── FDA label indication reverse scan ──
        # Pilocarpine / Cevimeline etc. are indexed under their drug name,
        # NOT the disease name. Query with broad disease-centric phrases
        # and keep only FDA labels whose indications text mentions the disease.
        fda_filt = Filter(must=[FieldCondition(
            key="section_name", match=MatchValue(value="indications_and_usage"))])
        fda_scan_queries = []
        for term in list(dict.fromkeys(t.lower() for t in search_terms))[:3]:
            fda_scan_queries.extend([
                ("FDA_Drug_Labels", f"{term} treatment symptoms medication drug", 50, fda_filt),
                ("FDA_Drug_Labels", f"{term} FDA approved indication therapy", 50, fda_filt),
            ])
        fda_raw = self._parallel_search(fda_scan_queries)
        fda_indicated_count = 0
        for key, hits in fda_raw.items():
            for h in hits:
                p = h["payload"]
                txt = self._ascii_fold(p.get("text_content", "")).lower()
                if not any(_disease_matches(a, txt) for a in _alias_set):
                    continue
                # Extract drug name from generic_name or brand_name
                gn = (p.get("generic_name") or "").strip()
                bn = (p.get("brand_name") or "").strip()
                dname = gn or bn
                if not dname:
                    continue
                # Normalise: take the first word of multi-salt names
                norm = dname.split()[0].upper() if dname else ""
                if not norm or len(norm) < 3:
                    continue
                # FDA label mentioning the disease → score at least 0.5
                # to ensure it survives the cap alongside gene-based drugs
                boosted_score = max(h["score"], 0.50)
                if norm not in drug_map:
                    drug_map[norm] = {
                        "drug_name": dname.split()[0].title(),
                        "source": "FDA_Drug_Labels",
                        "indication": disease_name,
                        "phase": 4,  # FDA-labelled ⇒ approved
                        "score": boosted_score,
                        "chembl_id": None,
                        "_primary": True,
                    }
                    fda_indicated_count += 1
                else:
                    drug_map[norm]["_primary"] = True
                    drug_map[norm]["score"] = max(drug_map[norm]["score"], boosted_score)
        if fda_indicated_count:
            logger.info(f"FDA indication scan: found {fda_indicated_count} new indicated drugs")

        # ── OT association disease-name scroll (catch-all for drugs with direct disease links) ──
        # Catches drugs like Mycophenolate whose gene targets (IMPDH) may not be in the
        # patient's DEG list but have strong OT disease associations.
        if "OpenTargets_data" in self._available:
            ot_assoc_count = 0
            try:
                ot_dis_filter = Filter(must=[
                    FieldCondition(key="entity_type", match=MatchValue(value="association")),
                ])
                self._queried_collections.add("OpenTargets_data")
                points, _ = self.client.scroll(
                    "OpenTargets_data", scroll_filter=ot_dis_filter,
                    limit=200, with_payload=True,
                )
                for pt in points:
                    p = pt.payload
                    dn = (p.get("disease_name") or "").lower()
                    if not any(_disease_matches(a, dn) for a in _alias_set):
                        continue
                    # This association is for our disease — extract the drug if possible
                    tn = (p.get("target_name") or "").strip()
                    if not tn:
                        continue
                    # target_name is a gene symbol; we need to find drugs targeting it
                    # The score tells us how strong the gene-disease link is
                    ot_score_val = None
                    for sf in (self._ot_score_fields or ["score"]):
                        ot_score_val = p.get(sf)
                        if ot_score_val is not None:
                            break
                    if ot_score_val is not None:
                        try:
                            ot_score_val = float(ot_score_val)
                        except (ValueError, TypeError):
                            ot_score_val = None
                    # Store gene-disease associations for downstream enrichment
                    # (drugs for these genes are discovered in Stage 2 enrichment)
            except Exception as e:
                logger.warning(f"OT association disease scroll failed: {e}")

        # Primary disease drugs retained first; alias-only drugs fill remaining slots
        ranked = sorted(drug_map.values(),
                        key=lambda x: (not x.get("_primary"), -x["score"]))
        cap = top_k * 5
        result = ranked[:cap]
        for r in result:
            r.pop("_primary", None)

        per_coll = {}
        for c in result:
            per_coll[c["source"]] = per_coll.get(c["source"], 0) + 1
        logger.info(f"find_drugs_for_disease({disease_name}): {per_coll}")
        return result

    def get_pathway_drugs(self, pathway_name: str, key_genes: Optional[List[str]] = None, top_k: int = 10) -> List[Dict]:
        """Find drugs targeting a pathway — queries Drug_agent + Raw_csv_KG."""
        queries = [
            ("Drug_agent", f"{pathway_name} drug target",
             top_k, Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="pathway_drug"))])),
            ("Raw_csv_KG", f"{pathway_name} drug treatment pathway", top_k, None),
        ]

        raw = self._parallel_search(queries)
        candidates = []
        seen = set()

        for key, hits in raw.items():
            coll = key.split("|")[0]
            for h in hits:
                p = h["payload"]
                drugs = self._extract_drug_names(p, coll)
                for dname in drugs:
                    norm = dname.upper().strip()
                    if norm and norm not in seen:
                        seen.add(norm)
                        candidates.append({
                            "drug_name": dname, "source": coll,
                            "pathway": pathway_name, "score": h["score"],
                        })

        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]

    # ── Stage 2: Evidence Enrichment ─────────────────────────────────────────

    def get_drug_identity(self, drug_name: str) -> Dict:
        """Merge identity from ChEMBL + FDA_DrugsFDA + OT_drugs_enriched + Orange Book."""
        queries = [
            ("ChEMBL_drugs", drug_name, 3, None),
            ("FDA_DrugsFDA", drug_name, 3, None),
            ("OpenTargets_drugs_enriched", drug_name, 3, None),
            ("FDA_Orange_Book", drug_name, 5, None),
        ]
        raw = self._parallel_search(queries)
        identity: Dict = {
            "drug_name": drug_name, "chembl_id": None, "drug_type": None,
            "max_phase": None, "first_approval": None, "is_fda_approved": False,
            "brand_names": [], "patent_count": 0, "exclusivity_count": 0,
            "generics_available": False, "pharm_class_moa": None,
            "pharm_class_epc": None, "withdrawn": False,
        }

        for key, hits in raw.items():
            coll = key.split("|")[0]
            for h in hits:
                p = h["payload"]
                name_match = self._fuzzy_drug_match(drug_name, p, coll)
                if not name_match:
                    continue

                if coll == "ChEMBL_drugs":
                    identity["chembl_id"] = identity["chembl_id"] or p.get("chembl_id")
                    identity["drug_type"] = identity["drug_type"] or p.get("molecule_type")
                    identity["max_phase"] = max(filter(None, [identity["max_phase"], p.get("max_phase")]), default=None)
                    identity["first_approval"] = identity["first_approval"] or p.get("first_approval")
                    if p.get("approval_status", "").upper().startswith("FDA"):
                        identity["is_fda_approved"] = True
                    # Capture ChEMBL canonical drug_name (often the INN)
                    chembl_dn = p.get("drug_name", "")
                    if chembl_dn and not identity.get("chembl_drug_name"):
                        identity["chembl_drug_name"] = chembl_dn
                    for syn in p.get("synonyms", []):
                        if syn and syn not in identity["brand_names"]:
                            identity["brand_names"].append(syn)

                elif coll == "FDA_DrugsFDA":
                    identity["is_fda_approved"] = True
                    identity["pharm_class_moa"] = identity["pharm_class_moa"] or p.get("pharm_class_moa")
                    identity["pharm_class_epc"] = identity["pharm_class_epc"] or p.get("pharm_class_epc")
                    for bn in [p.get("brand_name", "")]:
                        if bn and bn not in identity["brand_names"]:
                            identity["brand_names"].append(bn)

                elif coll == "OpenTargets_drugs_enriched":
                    identity["drug_type"] = identity["drug_type"] or p.get("drug_type")
                    identity["max_phase"] = max(filter(None, [identity["max_phase"], p.get("max_phase")]), default=None)
                    identity["withdrawn"] = identity["withdrawn"] or p.get("withdrawn", False)

                elif coll == "FDA_Orange_Book":
                    identity["patent_count"] = max(identity["patent_count"], p.get("patent_count", 0))
                    identity["exclusivity_count"] = max(identity["exclusivity_count"], p.get("exclusivity_count", 0))
                    if p.get("approval_date"):
                        identity["is_fda_approved"] = True
                    trade = p.get("trade_name", "")
                    if trade and trade not in identity["brand_names"]:
                        identity["brand_names"].append(trade)
                    if p.get("nda_type", "").upper() == "A":
                        identity["generics_available"] = True

        return identity

    def get_drug_targets(self, drug_name: str) -> List[Dict]:
        """Get target genes — ChEMBL + FDA Labels MoA + Drug_agent + Raw_csv_KG."""
        queries = [
            ("ChEMBL_drugs", drug_name, 5, None),
            ("FDA_Drug_Labels", drug_name, 3,
             Filter(must=[FieldCondition(key="section_name", match=MatchValue(value="mechanism_of_action"))])),
            ("Drug_agent", f"{drug_name} gene target", 5,
             Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="gene_drug"))])),
            ("Raw_csv_KG", f"{drug_name} target gene protein", 5, None),
        ]
        raw = self._parallel_search(queries)
        targets: Dict[str, Dict] = {}

        for key, hits in raw.items():
            coll = key.split("|")[0]
            for h in hits:
                p = h["payload"]
                if not self._fuzzy_drug_match(drug_name, p, coll):
                    continue

                if coll == "ChEMBL_drugs":
                    genes = p.get("target_gene_symbols", [])
                    actions = p.get("action_types", [])
                    moa = p.get("mechanism_of_action", "")
                    for i, gene in enumerate(genes):
                        gu = gene.upper()
                        if gu not in targets:
                            targets[gu] = {"gene_symbol": gene, "action_type": actions[i] if i < len(actions) else "UNKNOWN", "mechanism": moa}
                        else:
                            existing = targets[gu]["action_type"]
                            incoming = actions[i] if i < len(actions) else ""
                            if existing in ("", "UNKNOWN") and incoming not in ("", "UNKNOWN"):
                                targets[gu]["action_type"] = incoming

                elif coll == "FDA_Drug_Labels":
                    targets.setdefault("_FDA_MOA_", {})["fda_narrative"] = p.get("text_content", "")[:1000]

                elif coll == "Drug_agent":
                    gene = p.get("gene_symbol", "")
                    if gene:
                        gu = gene.upper()
                        if gu not in targets:
                            targets[gu] = {"gene_symbol": gene, "action_type": "UNKNOWN",
                                           "mechanism": p.get("mechanism_of_action", "")}

                elif coll == "Raw_csv_KG":
                    gene = self._kg_extract_gene(p)
                    if gene:
                        gu = gene.upper()
                        if gu not in targets:
                            targets[gu] = {"gene_symbol": gene, "action_type": "UNKNOWN",
                                           "mechanism": p.get("display_relation", p.get("relation", ""))}

        # Attach FDA MoA narrative to all targets
        fda_narrative = targets.pop("_FDA_MOA_", {}).get("fda_narrative", "")
        result = []
        for data in targets.values():
            data["fda_narrative"] = fda_narrative
            result.append(data)
        return result

    def get_target_disease_score(self, gene_symbol: str, disease_name: str) -> Optional[float]:
        """OpenTargets association score + KG supporting evidence.

        Uses Qdrant payload filters on target_name/disease_name for OT data
        (vector search fails because ~20K association records share identical
        template text).  Raw_csv_KG still uses semantic search.
        """
        disease_name = re.sub(r"\s*\(.*?\)", "", disease_name).strip()

        # Split compound disease names so each component is searched independently
        disease_components = [d.strip() for d in re.split(r"\s*/\s*|\s*,\s*|\s*\+\s*|\s+and\s+", disease_name) if d.strip()]
        if not disease_components:
            disease_components = [disease_name]

        best_score = None
        gene_upper = gene_symbol.upper()

        for component in disease_components:
            # ── OT: payload-filtered scroll (deterministic, no vector search) ──
            if "OpenTargets_data" in self._available:
                try:
                    ot_filter = Filter(must=[
                        FieldCondition(key="entity_type", match=MatchValue(value="association")),
                        FieldCondition(key="target_name", match=MatchValue(value=gene_upper)),
                    ])
                    self._queried_collections.add("OpenTargets_data")
                    points, _ = self.client.scroll(
                        "OpenTargets_data", scroll_filter=ot_filter,
                        limit=50, with_payload=True,
                    )
                    comp_lower = component.lower()
                    for pt in points:
                        p = pt.payload
                        dn = (p.get("disease_name") or "").lower()
                        if not _disease_matches(comp_lower, dn):
                            continue
                        # Extract score from probed fields
                        s = None
                        for sf in (self._ot_score_fields or ["score", "overall_score", "overall_association_score"]):
                            s = p.get(sf)
                            if s is not None:
                                break
                        if s is not None:
                            try:
                                s = float(s)
                                best_score = max(best_score or 0, s)
                            except (ValueError, TypeError):
                                pass
                except Exception as e:
                    logger.warning(f"OT payload-filter scroll failed for {gene_upper}+{component}: {e}")

            # ── Raw_csv_KG: semantic search (heterogeneous text, works well) ──
            if "Raw_csv_KG" in self._available:
                kg_queries = [("Raw_csv_KG", f"{gene_symbol} {component} associated", 3, None)]
                raw = self._parallel_search(kg_queries)
                for key, hits in raw.items():
                    for h in hits:
                        p = h["payload"]
                        s = None
                        for sf in (self._ot_score_fields or ["score", "overall_score", "overall_association_score"]):
                            s = p.get(sf)
                            if s is not None:
                                break
                        if s is not None:
                            try:
                                s = float(s)
                                best_score = max(best_score or 0, s)
                            except (ValueError, TypeError):
                                pass

        return best_score

    def get_indication_status(self, drug_name: str, disease_name: str,
                               disease_aliases: Optional[List[str]] = None) -> Dict:
        """Check if drug is approved for disease — FDA Labels + OT + Orange Book."""
        queries = [
            ("FDA_Drug_Labels", f"{drug_name} {disease_name}", 3,
             Filter(must=[FieldCondition(key="section_name", match=MatchValue(value="indications_and_usage"))])),
            ("OpenTargets_drugs_enriched", f"{drug_name} {disease_name}", 3, None),
            ("FDA_Orange_Book", drug_name, 3, None),
        ]
        raw = self._parallel_search(queries)
        result = {"is_approved": False, "highest_phase": None, "indication_text": "", "approval_date": None}
        alias_set = {self._ascii_fold(a).lower() for a in ([disease_name] + (disease_aliases or []))}

        for key, hits in raw.items():
            coll = key.split("|")[0]
            for h in hits:
                p = h["payload"]
                if not self._fuzzy_drug_match(drug_name, p, coll):
                    continue

                if coll == "FDA_Drug_Labels":
                    text = p.get("text_content", "")
                    text_lower = self._ascii_fold(text).lower()
                    if any(_disease_matches(alias, text_lower) for alias in alias_set):
                        result["is_approved"] = True
                        result["indication_text"] = text[:500]

                elif coll == "OpenTargets_drugs_enriched":
                    for ind in p.get("indications", []):
                        ind_name = self._ascii_fold(ind.get("disease_name", "")).lower()
                        if any(_disease_matches(alias, ind_name) or _disease_matches(ind_name, alias) for alias in alias_set):
                            result["is_approved"] = True
                            if not result["indication_text"]:
                                result["indication_text"] = ind.get("disease_name", "")
                            phase = ind.get("phase") or ind.get("clinical_phase")
                            if phase is not None:
                                try:
                                    result["highest_phase"] = max(result["highest_phase"] or 0, int(phase))
                                except (ValueError, TypeError):
                                    pass

                elif coll == "FDA_Orange_Book":
                    if p.get("approval_date"):
                        result["approval_date"] = p["approval_date"]
        return result

    def get_trial_evidence(self, drug_name: str, disease_name: str,
                            disease_aliases: Optional[List[str]] = None) -> Dict:
        """Clinical trial data — summaries + results."""
        import numpy as np
        queries = [
            ("ClinicalTrials_results", f"{drug_name} {disease_name}", 15, None),
            ("ClinicalTrials_summaries", f"{drug_name} {disease_name}", 15, None),
        ]
        raw = self._parallel_search(queries)
        drug_upper = drug_name.upper()
        _GENERIC = {"cancer", "disease", "syndrome", "chronic", "acute", "primary", "advanced", "metastatic", "stage", "type", "with", "cell"}
        all_names = [disease_name] + (disease_aliases or [])
        disease_keywords = list({self._ascii_fold(w).lower() for name in all_names
                                 for w in name.split() if len(w) >= 4 and w.lower() not in _GENERIC})
        disease_vec = None
        evidence = {
            "total_trials": 0, "highest_phase": None, "completed_trials": 0,
            "trials_with_results": 0, "best_p_value": None, "total_enrollment": 0,
            "top_trials": [],
        }
        seen_ncts = set()

        for key, hits in raw.items():
            coll = key.split("|")[0]
            for h in hits:
                p = h["payload"]
                trial_text = (p.get("brief_title", "") + " " + p.get("text_content", "")[:500]).upper()

                # Verify the trial actually involves this drug
                trial_drugs = [d.upper() for d in p.get("drug_names", [])]
                if trial_drugs:
                    if not any(drug_upper in d or d in drug_upper for d in trial_drugs):
                        continue
                else:
                    if drug_upper not in trial_text:
                        continue

                # Verify trial is relevant to the target disease
                # Pass 1: keyword match
                disease_relevant = (
                    not disease_keywords
                    or any(kw.upper() in trial_text for kw in disease_keywords)
                )
                # Pass 2: semantic fallback when keywords miss synonyms
                #         (e.g. "myelogenous" vs "myeloid")
                if not disease_relevant and self.embedder:
                    try:
                        if disease_vec is None:
                            disease_vec = self.embedder.encode(disease_name)
                        title = p.get("brief_title", "")[:200]
                        title_vec = self.embedder.encode(title)
                        sim = float(np.dot(
                            disease_vec / (np.linalg.norm(disease_vec) + 1e-10),
                            title_vec / (np.linalg.norm(title_vec) + 1e-10),
                        ))
                        if sim >= 0.6:
                            disease_relevant = True
                    except Exception:
                        pass
                if not disease_relevant:
                    continue

                nct = p.get("nct_id", "")
                if nct in seen_ncts:
                    continue
                seen_ncts.add(nct)

                evidence["total_trials"] += 1
                phase_num = p.get("phase_numeric")
                if phase_num is not None:
                    try:
                        evidence["highest_phase"] = max(evidence["highest_phase"] or 0, float(phase_num))
                    except (ValueError, TypeError):
                        pass

                status = p.get("overall_status", "").upper()
                if "COMPLETED" in status:
                    evidence["completed_trials"] += 1

                enrollment = p.get("enrollment")
                if enrollment:
                    try:
                        evidence["total_enrollment"] += int(enrollment)
                    except (ValueError, TypeError):
                        pass

                if coll == "ClinicalTrials_results":
                    evidence["trials_with_results"] += 1
                    for pv in p.get("p_values", []):
                        try:
                            pv_f = float(pv)
                            if evidence["best_p_value"] is None or pv_f < evidence["best_p_value"]:
                                evidence["best_p_value"] = pv_f
                        except (ValueError, TypeError):
                            pass

                if len(evidence["top_trials"]) < 5:
                    why = p.get("why_stopped", "") or ""
                    evidence["top_trials"].append({
                        "nct_id": nct, "title": p.get("brief_title", ""),
                        "phase": p.get("phase", ""), "status": p.get("overall_status", ""),
                        "has_results": coll == "ClinicalTrials_results",
                        "why_stopped": why,
                    })

        _SAFETY_TERMS = {"safety", "toxicity", "adverse", "death", "fatal", "harmful"}
        evidence["stopped_for_safety"] = any(
            any(term in t.get("why_stopped", "").lower() for term in _SAFETY_TERMS)
            for t in evidence["top_trials"]
        )
        return evidence

    def get_safety_profile(self, drug_name: str) -> Dict:
        """Aggregate safety — FDA Labels (3 sections) + OT AEs + FAERS + PGx + Enforcement."""
        queries = [
            ("FDA_Drug_Labels", drug_name, 3,
             Filter(must=[FieldCondition(key="section_name", match=MatchValue(value="boxed_warning"))])),
            ("FDA_Drug_Labels", drug_name, 5,
             Filter(must=[FieldCondition(key="section_name", match=MatchValue(value="adverse_reactions"))])),
            ("FDA_Drug_Labels", drug_name, 3,
             Filter(must=[FieldCondition(key="section_name", match=MatchValue(value="contraindications"))])),
            ("OpenTargets_adverse_events", drug_name, 10, None),
            ("FDA_FAERS", drug_name, 5, None),
            ("OpenTargets_pharmacogenomics", drug_name, 5, None),
            ("FDA_Enforcement", drug_name, 5, None),
        ]
        raw = self._parallel_search(queries)
        profile: Dict = {
            "boxed_warnings": [], "top_adverse_events": [],
            "serious_ratio": None, "fatal_ratio": None,
            "contraindications": [], "pgx_warnings": [], "recall_history": [],
        }

        for key, hits in raw.items():
            coll = key.split("|")[0]
            for h in hits:
                p = h["payload"]
                if not self._fuzzy_drug_match(drug_name, p, coll):
                    continue

                if coll == "FDA_Drug_Labels":
                    section = p.get("section_name", "")
                    text = p.get("text_content", "")[:500]
                    if section == "boxed_warning" and text:
                        profile["boxed_warnings"].append(text)
                    elif section == "adverse_reactions" and text:
                        profile["top_adverse_events"].append({"text": text, "source": "FDA_Label"})
                    elif section == "contraindications" and text:
                        profile["contraindications"].append(text)

                elif coll == "OpenTargets_adverse_events":
                    profile["top_adverse_events"].append({
                        "event_name": p.get("event_name", ""),
                        "log_lr": p.get("log_lr"), "count": p.get("report_count"),
                        "source": "OpenTargets",
                    })

                elif coll == "FDA_FAERS":
                    if p.get("entity_type") == "faers_summary":
                        try:
                            profile["serious_ratio"] = float(p["serious_pct"]) / 100 if p.get("serious_pct") else profile["serious_ratio"]
                            profile["fatal_ratio"] = float(p["fatal_pct"]) / 100 if p.get("fatal_pct") else profile["fatal_ratio"]
                        except (ValueError, TypeError):
                            pass
                    else:
                        profile["top_adverse_events"].append({
                            "event_name": p.get("reaction_term", ""),
                            "count": p.get("reaction_count"), "source": "FAERS",
                        })

                elif coll == "OpenTargets_pharmacogenomics":
                    profile["pgx_warnings"].append({
                        "gene": p.get("gene_symbol", ""),
                        "variant": p.get("variant_rs_id", ""),
                        "phenotype": p.get("phenotype_text", ""),
                        "category": p.get("pgx_category", ""),
                        "evidence_level": p.get("evidence_level", ""),
                    })

                elif coll == "FDA_Enforcement":
                    profile["recall_history"].append({
                        "classification": p.get("classification", ""),
                        "reason": p.get("reason_for_recall", "")[:200],
                        "status": p.get("status", ""),
                        "date": p.get("recall_initiation_date", ""),
                    })

        # Sort AEs by log_lr where available
        profile["top_adverse_events"].sort(
            key=lambda x: x.get("log_lr") or 0, reverse=True)
        return profile

    def _get_disease_ae_synonyms(self, disease_name: str) -> set:
        """Dynamically discover MedDRA-style synonyms for a disease by querying the AE collection."""
        cache_key = f"ae_syn:{disease_name}"
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        import numpy as np
        disease_clean = re.sub(r"\s*\(.*?\)", "", disease_name).strip()
        synonyms = {disease_clean.lower()}

        hits = self._search("OpenTargets_adverse_events", disease_clean, 15)
        if hits and self.embedder:
            d_vec = self.embedder.encode(disease_clean)
            d_norm = d_vec / (np.linalg.norm(d_vec) + 1e-10)
            for h in hits:
                term = h["payload"].get("event_name", "")
                if not term or len(term) < 4:
                    continue
                t_vec = self.embedder.encode(term)
                sim = float(np.dot(d_norm, t_vec / (np.linalg.norm(t_vec) + 1e-10)))
                if sim >= 0.55:
                    synonyms.add(term.lower())

        self._embed_cache[cache_key] = synonyms
        return synonyms

    def get_disease_aliases(self, disease_name: str) -> List[str]:
        """Dynamically discover disease synonyms from OT disease entities + drug indication names."""
        import numpy as np
        cache_key = f"disease_aliases:{disease_name}"
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        d_vec = self.embedder.encode(disease_name)
        d_norm = d_vec / (np.linalg.norm(d_vec) + 1e-10)
        candidate_names: set = set()

        # Source 1: OT disease entities
        filt = Filter(must=[FieldCondition(key="entity_type", match=MatchValue(value="disease"))])
        for h in self._search("OpenTargets_data", disease_name, 30, filt):
            name = h["payload"].get("name", "")
            if name and len(name) >= 4:
                candidate_names.add(name)

        # Source 2: indication disease_names from drugs semantically near this disease
        for h in self._search("OpenTargets_drugs_enriched", disease_name, 20):
            for ind in h["payload"].get("indications", []):
                name = ind.get("disease_name", "")
                if name and len(name) >= 4:
                    candidate_names.add(name)

        disease_lower = disease_name.lower()
        _STOPWORDS = {"syndrome", "disease", "disorder", "condition", "primary", "secondary", "chronic", "acute", "of", "the", "and", "in", "with"}
        disease_content = {self._ascii_fold(w).lower().rstrip("'s") for w in disease_name.split()
                           if len(w) >= 3 and w.lower().rstrip("'s") not in _STOPWORDS}
        scored = []
        for name in candidate_names:
            if name.lower() == disease_lower:
                continue
            t_vec = self.embedder.encode(name)
            sim = float(np.dot(d_norm, t_vec / (np.linalg.norm(t_vec) + 1e-10)))
            if sim <= 0.60:
                continue
            # Reject aliases that share zero content words with the original disease
            alias_content = {self._ascii_fold(w).lower().rstrip("'s") for w in name.split()
                             if len(w) >= 3 and w.lower().rstrip("'s") not in _STOPWORDS}
            if disease_content and alias_content and not disease_content & alias_content:
                continue
            scored.append((sim, name))

        scored.sort(reverse=True)
        aliases = [disease_name] + [name for _, name in scored[:12]]

        self._embed_cache[cache_key] = aliases
        logger.info(f"Disease aliases for '{disease_name}': {aliases}")
        print(f"        Disease aliases for '{disease_name}': {aliases}")
        return aliases

    def check_disease_in_adverse_events(self, safety: Dict, disease_name: str) -> Dict:
        """Cross-reference a drug's adverse events against the patient's disease.
        Catches drugs that CAUSE the condition being treated."""
        import numpy as np
        disease_clean = re.sub(r"\s*\(.*?\)", "", disease_name).strip()
        disease_synonyms = self._get_disease_ae_synonyms(disease_name)
        disease_tokens_all = {w for syn in disease_synonyms for w in syn.split() if len(w) >= 4}
        matching = []

        for ae in safety.get("top_adverse_events", []):
            event = ae.get("event_name", "").lower()
            if not event:
                continue
            # Direct match against any synonym
            if any(syn in event or event in syn for syn in disease_synonyms):
                matching.append(ae.get("event_name", event))
                continue
            # Token overlap across all synonyms
            event_tokens = {w for w in event.split() if len(w) >= 4}
            if disease_tokens_all and event_tokens and len(disease_tokens_all & event_tokens) >= max(1, len(disease_tokens_all) // 2):
                matching.append(ae.get("event_name", event))
                continue
            # Semantic fallback
            if self.embedder:
                try:
                    d_vec = self.embedder.encode(disease_clean)
                    e_vec = self.embedder.encode(event)
                    sim = float(np.dot(
                        d_vec / (np.linalg.norm(d_vec) + 1e-10),
                        e_vec / (np.linalg.norm(e_vec) + 1e-10),
                    ))
                    if sim >= 0.50:
                        matching.append(f"{ae.get('event_name', event)} (semantic={sim:.2f})")
                except Exception:
                    pass

        # Scan FDA label text (boxed warnings + contraindications)
        for field_key in ("boxed_warnings", "contraindications"):
            for text in safety.get(field_key, []):
                text_lower = text.lower()
                if any(syn in text_lower for syn in disease_synonyms):
                    matching.append(f"FDA label {field_key}: mentions {disease_clean}")

        if matching:
            return {
                "is_contraindicated": True,
                "reason": f"Drug's adverse events include the patient's disease ({', '.join(matching[:3])})",
                "matching_events": matching,
            }
        return {"is_contraindicated": False, "reason": "", "matching_events": []}

    def check_contraindication(self, drug_name: str, gene_symbol: str,
                                gene_direction: str, log2fc: float = 0.0) -> Dict:
        """Check if drug's action on target conflicts with patient gene expression.
        Returns tier: 2 (Contraindicated) when |log2fc| >= clinical threshold,
                      3 (Use With Caution) when below threshold.
        """
        from agentic_ai_wf.reporting_pipeline_agent.core_types import DEG_LOG2FC_THRESHOLD
        results = self._search("ChEMBL_drugs", f"{drug_name} {gene_symbol}", 5)
        gene_up = gene_symbol.upper()

        for h in results:
            p = h["payload"]
            if not self._fuzzy_drug_match(drug_name, p, "ChEMBL_drugs"):
                continue
            targets = [t.upper() for t in p.get("target_gene_symbols", [])]
            actions = p.get("action_types", [])
            if gene_up not in targets:
                continue

            idx = targets.index(gene_up)
            action = actions[idx].upper() if idx < len(actions) else ""
            tier = 2 if abs(log2fc) >= DEG_LOG2FC_THRESHOLD else 3

            if gene_direction == "down" and action in ("INHIBITOR", "NEGATIVE MODULATOR", "ANTAGONIST", "BLOCKER"):
                return {"is_contraindicated": True,
                        "reason": f"{drug_name} is a {action} of {gene_symbol} which is already downregulated (inhibiting a suppressed target)",
                        "severity": "high", "tier": tier}
            if gene_direction == "up" and action in ("AGONIST", "POSITIVE MODULATOR", "ACTIVATOR"):
                return {"is_contraindicated": True,
                        "reason": f"{drug_name} is a {action} of {gene_symbol} which is upregulated (may reinforce oncogenic driver)",
                        "severity": "moderate", "tier": tier}

        return {"is_contraindicated": False, "reason": "", "severity": "", "tier": 0}

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _extract_drug_names(self, payload: Dict, collection: str) -> List[str]:
        """Dynamically extract drug names from any collection's payload."""
        names = []
        # Direct drug name fields
        for field in ("drug_name", "name", "molecule_name", "generic_name",
                       "ingredient", "brand_name", "trade_name"):
            val = payload.get(field, "")
            if val and isinstance(val, str) and len(val) > 1:
                names.append(val)

        # List fields containing drug names
        for field in ("drug_names", "synonyms"):
            vals = payload.get(field, [])
            if isinstance(vals, list):
                names.extend([v for v in vals if isinstance(v, str) and len(v) > 1])

        # Nested drug lists in Drug_agent disease_drug docs
        for entry in payload.get("approved_drugs", []):
            if isinstance(entry, dict) and entry.get("drug_name"):
                names.append(entry["drug_name"])

        # Nested drug lists in Drug_agent pathway_drug docs
        for entry in payload.get("targeting_drugs", []):
            if isinstance(entry, dict) and entry.get("drug_name"):
                names.append(entry["drug_name"])

        # KG triples — extract the Drug entity
        if collection == "Raw_csv_KG":
            for side in ("x", "y"):
                if payload.get(f"{side}_type", "").lower() in ("drug", "compound"):
                    n = payload.get(f"{side}_name", "")
                    if n:
                        names.append(n)

        return list(dict.fromkeys(names))  # Deduplicate preserving order

    def _extract_action_type(self, payload: Dict, collection: str) -> str:
        if collection == "ChEMBL_drugs":
            actions = payload.get("action_types", [])
            return actions[0] if actions else "UNKNOWN"
        return payload.get("action_type", "UNKNOWN")

    def _fuzzy_drug_match(self, query_drug: str, payload: Dict, collection: str) -> bool:
        """Check if payload plausibly refers to the queried drug."""
        q = query_drug.upper().strip()
        candidates = self._extract_drug_names(payload, collection)
        return any(q in c.upper() or c.upper() in q for c in candidates) if candidates else True

    def _kg_involves_entity(self, payload: Dict, entity_name: str, entity_types: set) -> bool:
        """Check if a KG triple involves a specific entity type."""
        for side in ("x", "y"):
            etype = payload.get(f"{side}_type", "").lower()
            if etype in entity_types:
                if not entity_name:
                    return True
                if entity_name.upper() in payload.get(f"{side}_name", "").upper():
                    return True
        return False

    def _kg_extract_gene(self, payload: Dict) -> Optional[str]:
        for side in ("x", "y"):
            if payload.get(f"{side}_type", "").lower() in ("gene", "protein"):
                return payload.get(f"{side}_name")
        return None

    def shutdown(self):
        self._executor.shutdown(wait=False)
