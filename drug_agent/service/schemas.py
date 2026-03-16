"""Service contract — input/output dataclasses for the Drug Agent Service."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

try:
    from agentic_ai_wf.reporting_pipeline_agent.core_types import (
        DRUG_HIGH_PRIORITY_THRESHOLD, DRUG_MODERATE_PRIORITY_THRESHOLD,
    )
except ImportError:
    DRUG_HIGH_PRIORITY_THRESHOLD, DRUG_MODERATE_PRIORITY_THRESHOLD = 55, 30


class QueryType(Enum):
    FULL_RECOMMENDATION = "full_recommendation"
    VALIDATE_DRUG = "validate_drug"
    CHECK_CONTRAINDICATION = "check_contraindication"
    SAFETY_PROFILE = "safety_profile"
    DRUG_DETAILS = "drug_details"


# ── Request Context ──────────────────────────────────────────────────────────

@dataclass
class GeneContext:
    gene_symbol: str
    log2fc: float
    adj_p_value: float
    direction: str                                    # "up" | "down"
    role: Optional[str] = None                        # "pathogenic" | "protective" | "therapeutic_target" | "immune_modulator"
    composite_score: Optional[float] = None
    evidence_stratum: Optional[str] = None            # "known_driver" | "ppi_connected" | "expression_significant" | "novel_candidate"
    causal_tier: Optional[str] = None

@dataclass
class PathwayContext:
    pathway_name: str
    direction: str
    fdr: float
    gene_count: int
    category: Optional[str] = None
    key_genes: Optional[List[str]] = None
    disease_relevance: Optional[str] = None           # ME-validated therapeutic implication

@dataclass
class BiomarkerContext:
    biomarker_name: str
    status: str                                       # "positive" | "negative" | "not_assessed" | "suggestive"
    supporting_genes: Optional[List[str]] = None
    biomarker_type: Optional[str] = None              # "A" (RNA-assessable) | "B" (orthogonal test required)

@dataclass
class TMEContext:
    highly_enriched_cells: List[str] = field(default_factory=list)
    moderately_enriched_cells: List[str] = field(default_factory=list)
    immune_infiltration_level: str = "unknown"

@dataclass
class MolecularSignatures:
    proliferation: Optional[float] = None
    apoptosis: Optional[float] = None
    dna_repair: Optional[float] = None
    inflammation: Optional[float] = None
    immune_activation: Optional[float] = None


@dataclass
class ScoringConfig:
    target_direction_weight: float = 18.0
    target_magnitude_weight: float = 12.0
    clinical_regulatory_weight: float = 25.0
    ot_weight: float = 15.0
    pathway_weight: float = 15.0
    safety_max_penalty: float = -30.0
    signature_bonus_max: float = 8.0
    # Fix 2: tier-weighted contra multiplier
    apply_contra_multipliers: bool = True
    contra_tier_multipliers: Dict = field(default_factory=lambda: {1: 0.0, 2: 0.25, 3: 0.75})
    # Fix 5: SOC multi-signal composite
    use_soc_composite: bool = True
    soc_signal_weights: Dict = field(default_factory=lambda: {
        "indication_sim": 0.40, "pharm_class_sim": 0.25, "clinical_depth": 0.35,
    })
    # Gene evidence stratum multipliers (known_driver=full credit → novel_candidate=half)
    stratum_multipliers: Dict = field(default_factory=lambda: {
        "known_driver": 1.0, "ppi_connected": 0.85,
        "expression_significant": 0.65, "novel_candidate": 0.5,
    })
    # Downstream effector analysis thresholds
    min_effectors_concordant: int = 2
    effector_credit_fraction: float = 0.6
    # Stage 5 post-score filtering thresholds
    base_noise_threshold: float = 10.0
    high_clinical_noise_threshold: float = 5.0
    high_clinical_score_cutoff: float = 15.0


@dataclass
class DrugQueryRequest:
    disease: str
    query_type: QueryType = QueryType.FULL_RECOMMENDATION
    genes: List[GeneContext] = field(default_factory=list)
    pathways: List[PathwayContext] = field(default_factory=list)
    biomarkers: List[BiomarkerContext] = field(default_factory=list)
    tme: Optional[TMEContext] = None
    signatures: Optional[MolecularSignatures] = None
    drug_name: Optional[str] = None
    max_results: int = 30
    include_safety: bool = True
    include_trials: bool = True
    scoring_config: Optional[ScoringConfig] = None
    disease_context: Optional[str] = None             # Pipeline ME synthesis for OT fallback
    disease_aliases: List[str] = field(default_factory=list)
    all_patient_genes: List[GeneContext] = field(default_factory=list)  # Full DEG list for scoring (discovery uses `genes`)
    signature_scores: Optional[Dict] = None            # Full pathway signature scores (e.g., ifn, inflammation)

    def get_upregulated_targets(self) -> List[GeneContext]:
        return [g for g in self.genes if g.direction == "up"
                and g.role in ("pathogenic", "therapeutic_target", "immune_modulator", None)]

    def get_downregulated_genes(self) -> List[GeneContext]:
        return [g for g in self.genes if g.direction == "down"]

    def get_downregulated_genes_significant(self) -> List[GeneContext]:
        """Downregulated genes meeting clinical significance threshold (|log2FC| >= 0.58)."""
        from agentic_ai_wf.reporting_pipeline_agent.core_types import DEG_LOG2FC_THRESHOLD
        return [g for g in self.genes
                if g.direction == "down" and abs(g.log2fc) >= DEG_LOG2FC_THRESHOLD]


# ── Response Evidence ────────────────────────────────────────────────────────

@dataclass
class DrugIdentity:
    drug_name: str
    chembl_id: Optional[str] = None
    drug_type: Optional[str] = None
    max_phase: Optional[int] = None
    first_approval: Optional[int] = None
    is_fda_approved: bool = False
    brand_names: List[str] = field(default_factory=list)
    patent_count: int = 0
    exclusivity_count: int = 0
    generics_available: bool = False
    pharm_class_moa: Optional[str] = None
    pharm_class_epc: Optional[str] = None
    indication_text: Optional[str] = None
    withdrawn: bool = False
    genetic_eligibility_required: bool = False
    genetic_eligibility_detail: str = ""

@dataclass
class TargetEvidence:
    gene_symbol: str
    action_type: str
    mechanism_of_action: Optional[str] = None
    fda_moa_narrative: Optional[str] = None
    patient_gene_log2fc: Optional[float] = None
    patient_gene_direction: Optional[str] = None
    ot_association_score: Optional[float] = None
    related_patient_gene: Optional[str] = None        # Pathway co-member from patient DEGs
    related_gene_log2fc: Optional[float] = None
    related_gene_direction: Optional[str] = None
    related_gene_source: Optional[str] = None         # "pathway" | "knowledge_graph"
    downstream_effector_genes: Optional[List[str]] = None   # All dysregulated pathway members
    downstream_pathway: Optional[str] = None
    known_effectors: Optional[List[str]] = None            # KG-resolved functionally related genes

@dataclass
class TrialEvidence:
    total_trials: int = 0
    highest_phase: Optional[float] = None
    completed_trials: int = 0
    trials_with_results: int = 0
    best_p_value: Optional[float] = None
    total_enrollment: int = 0
    top_trials: List[Dict] = field(default_factory=list)
    stopped_for_safety: bool = False

@dataclass
class SafetyProfile:
    boxed_warnings: List[str] = field(default_factory=list)
    top_adverse_events: List[Dict] = field(default_factory=list)
    serious_ratio: Optional[float] = None
    fatal_ratio: Optional[float] = None
    contraindications: List[str] = field(default_factory=list)
    pgx_warnings: List[Dict] = field(default_factory=list)
    recall_history: List[Dict] = field(default_factory=list)

@dataclass
class ScoreBreakdown:
    target_direction_match: float = 0.0
    target_magnitude_match: float = 0.0
    clinical_regulatory_score: float = 0.0
    ot_association_score: float = 0.0
    pathway_concordance: float = 0.0
    safety_penalty: float = 0.0
    disease_indication_bonus: float = 0.0
    signature_bonus: float = 0.0
    gene_evidence_quality: float = 1.0                 # Stratum multiplier of best-matched gene (1.0=driver, 0.5=novel)
    composite_score: float = 0.0
    pipeline_evidence_used: bool = False               # True when OT fallback used pipeline context
    disease_relevant: bool = True                       # False when drug targets genes but lacks disease-treatment evidence
    tier_reasoning: str = ""                             # Q1→Q2→Q3 decision-tree explanation of tier placement

    def calculate(self):
        self.composite_score = max(0.0, min(100.0,
            self.target_direction_match
            + self.target_magnitude_match
            + self.clinical_regulatory_score
            + self.ot_association_score
            + self.pathway_concordance
            + self.safety_penalty
            + self.disease_indication_bonus
            + self.signature_bonus
        ))

@dataclass
class ContraindicationEntry:
    tier: int                                         # 1=Avoid, 2=Contraindicated, 3=Use With Caution
    reason: str
    source: str                                       # "gene_based" | "biomarker" | "disease_ae" | "trial_stopped" | "withdrawn"
    gene_symbol: Optional[str] = None
    log2fc: Optional[float] = None

    @property
    def label(self) -> str:
        return {1: "Avoid", 2: "Contraindicated", 3: "Use With Caution"}.get(self.tier, "Unknown")

@dataclass
class DrugCandidate:
    identity: DrugIdentity
    targets: List[TargetEvidence] = field(default_factory=list)
    trial_evidence: Optional[TrialEvidence] = None
    safety: Optional[SafetyProfile] = None
    score: Optional[ScoreBreakdown] = None
    contraindication_flags: List[str] = field(default_factory=list)
    contraindication_entries: List[ContraindicationEntry] = field(default_factory=list)
    caution_notes: List[ContraindicationEntry] = field(default_factory=list)
    is_soc_candidate: bool = False
    soc_confidence: float = 0.0
    soc_advisory_notes: List[str] = field(default_factory=list)
    evidence_sources: List[str] = field(default_factory=list)
    discovery_paths: List[str] = field(default_factory=list)
    validation_caveat: str = ""

@dataclass
class DrugQueryResponse:
    success: bool
    disease: str
    query_type: str
    recommendations: List[DrugCandidate] = field(default_factory=list)
    contraindicated: List[DrugCandidate] = field(default_factory=list)
    gene_targeted_only: List[DrugCandidate] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def high_priority(self) -> List[DrugCandidate]:
        return [r for r in self.recommendations if r.score and r.score.composite_score >= DRUG_HIGH_PRIORITY_THRESHOLD]

    @property
    def moderate_priority(self) -> List[DrugCandidate]:
        return [r for r in self.recommendations
                if r.score and DRUG_MODERATE_PRIORITY_THRESHOLD <= r.score.composite_score < DRUG_HIGH_PRIORITY_THRESHOLD]

    def to_dict(self) -> Dict:
        from dataclasses import asdict
        return asdict(self)
