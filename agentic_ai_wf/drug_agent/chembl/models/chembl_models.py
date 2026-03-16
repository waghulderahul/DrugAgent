"""
ChEMBL Data Models
==================

Pydantic/dataclass models for ChEMBL drug data.
These models define the schema for normalized ChEMBL data and Qdrant storage.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
import uuid


class ApprovalPhase(Enum):
    """Drug approval phases from ChEMBL."""
    APPROVED = 4
    PHASE_3 = 3
    PHASE_2 = 2
    PHASE_1 = 1
    EARLY_PHASE_1 = 0.5
    PRECLINICAL = 0
    UNKNOWN = -1


class ActionType(str, Enum):
    """Drug action types from ChEMBL mechanisms."""
    INHIBITOR = "INHIBITOR"
    ANTAGONIST = "ANTAGONIST"
    AGONIST = "AGONIST"
    BLOCKER = "BLOCKER"
    MODULATOR = "MODULATOR"
    ACTIVATOR = "ACTIVATOR"
    OPENER = "OPENER"
    BINDER = "BINDER"
    SUBSTRATE = "SUBSTRATE"
    COFACTOR = "COFACTOR"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"
    INDIRECT_EFFECT = "INDIRECT_EFFECT"


@dataclass
class TargetComponent:
    """Target component with gene symbol information."""
    accession: str = ""           # UniProt accession (e.g., P35367)
    component_type: str = ""      # PROTEIN, etc.
    gene_symbol: str = ""         # Gene symbol (e.g., HRH1) - KEY FIELD
    component_id: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetComponent":
        """Create from ChEMBL API response."""
        return cls(
            accession=data.get("accession", ""),
            component_type=data.get("component_type", ""),
            gene_symbol=data.get("gene_symbol", ""),
            component_id=data.get("component_id", 0)
        )


@dataclass
class MechanismInfo:
    """Mechanism of action information from ChEMBL."""
    mechanism_of_action: str = ""     # "Histamine H1 receptor antagonist"
    action_type: str = ""             # "ANTAGONIST"
    target_chembl_id: str = ""        # "CHEMBL231"
    target_name: str = ""             # "Histamine H1 receptor"
    target_type: str = ""             # "SINGLE PROTEIN"
    target_organism: str = ""         # "Homo sapiens"
    direct_interaction: bool = True
    disease_efficacy: bool = True
    
    # Gene linking (critical field)
    target_gene_symbols: List[str] = field(default_factory=list)  # ["HRH1"]
    target_uniprot_ids: List[str] = field(default_factory=list)   # ["P35367"]
    
    # Target components
    target_components: List[TargetComponent] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MechanismInfo":
        """Create from ChEMBL API response."""
        # Extract gene symbols from target_components (if enriched)
        components = []
        gene_symbols = []
        uniprot_ids = []
        
        # First check if target_gene_symbols was already extracted by MechanismFetcher
        if data.get("target_gene_symbols"):
            gene_symbols.extend(data.get("target_gene_symbols", []))
        
        for comp_data in data.get("target_components", []):
            comp = TargetComponent.from_dict(comp_data)
            components.append(comp)
            if comp.gene_symbol and comp.gene_symbol not in gene_symbols:
                gene_symbols.append(comp.gene_symbol)
            if comp.accession:
                uniprot_ids.append(comp.accession)
            
            # Also check synonyms for GENE_SYMBOL type
            for syn in comp_data.get("target_component_synonyms", []):
                if syn.get("syn_type") == "GENE_SYMBOL":
                    gene = syn.get("component_synonym")
                    if gene and gene not in gene_symbols:
                        gene_symbols.append(gene)
        
        return cls(
            mechanism_of_action=data.get("mechanism_of_action", ""),
            action_type=data.get("action_type", "UNKNOWN"),
            target_chembl_id=data.get("target_chembl_id", ""),
            target_name=data.get("target_name", ""),
            target_type=data.get("target_type", ""),
            target_organism=data.get("target_organism", ""),
            direct_interaction=data.get("direct_interaction", True),
            disease_efficacy=data.get("disease_efficacy", True),
            target_gene_symbols=gene_symbols,
            target_uniprot_ids=uniprot_ids,
            target_components=components
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "mechanism_of_action": self.mechanism_of_action,
            "action_type": self.action_type,
            "target_chembl_id": self.target_chembl_id,
            "target_name": self.target_name,
            "target_type": self.target_type,
            "target_organism": self.target_organism,
            "direct_interaction": self.direct_interaction,
            "disease_efficacy": self.disease_efficacy,
            "target_gene_symbols": self.target_gene_symbols,
            "target_uniprot_ids": self.target_uniprot_ids
        }


@dataclass
class IndicationInfo:
    """Drug indication (disease) information from ChEMBL."""
    mesh_id: str = ""              # "D006255"
    mesh_heading: str = ""         # "Rhinitis, Allergic, Seasonal"
    efo_id: str = ""               # "EFO_0003931"
    efo_term: str = ""             # "allergic rhinitis"
    max_phase_for_ind: int = 0     # 4 (approved for this indication)
    
    # Additional reference information
    indication_refs: List[Dict[str, str]] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndicationInfo":
        """Create from ChEMBL API response."""
        return cls(
            mesh_id=data.get("mesh_id", "") or "",
            mesh_heading=data.get("mesh_heading", "") or "",
            efo_id=data.get("efo_id", "") or "",
            efo_term=data.get("efo_term", "") or "",
            max_phase_for_ind=data.get("max_phase_for_ind", 0) or 0,
            indication_refs=data.get("indication_refs", []) or []
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "mesh_id": self.mesh_id,
            "mesh_heading": self.mesh_heading,
            "efo_id": self.efo_id,
            "efo_term": self.efo_term,
            "max_phase_for_ind": self.max_phase_for_ind
        }
    
    @property
    def disease_name(self) -> str:
        """Get best available disease name."""
        return self.mesh_heading or self.efo_term or ""


@dataclass
class MoleculeProperties:
    """Chemical properties of a molecule."""
    molecular_weight: Optional[float] = None
    alogp: Optional[float] = None          # Lipophilicity
    hba: Optional[int] = None              # H-bond acceptors
    hbd: Optional[int] = None              # H-bond donors
    psa: Optional[float] = None            # Polar surface area
    rtb: Optional[int] = None              # Rotatable bonds
    ro3_pass: Optional[str] = None         # Rule of 3 pass
    num_ro5_violations: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MoleculeProperties":
        """Create from ChEMBL API response."""
        if not data:
            return cls()
        return cls(
            molecular_weight=float(data.get("full_mwt") or data.get("mw_freebase") or 0) or None,
            alogp=float(data.get("alogp") or 0) or None,
            hba=int(data.get("hba") or 0) or None,
            hbd=int(data.get("hbd") or 0) or None,
            psa=float(data.get("psa") or 0) or None,
            rtb=int(data.get("rtb") or 0) or None,
            ro3_pass=data.get("ro3_pass"),
            num_ro5_violations=int(data.get("num_ro5_violations") or 0) or None
        )


@dataclass
class ChEMBLMolecule:
    """Raw molecule data from ChEMBL API."""
    molecule_chembl_id: str          # "CHEMBL998"
    pref_name: str = ""              # "LORATADINE"
    max_phase: float = 0             # 4
    molecule_type: str = ""          # "Small molecule"
    first_approval: Optional[int] = None  # 1993
    
    # Administration routes
    oral: bool = False
    parenteral: bool = False
    topical: bool = False
    
    # Synonyms
    molecule_synonyms: List[Dict[str, str]] = field(default_factory=list)
    
    # Chemical properties
    molecule_properties: Optional[MoleculeProperties] = None
    
    # Structure (optional)
    molecule_structures: Optional[Dict[str, str]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChEMBLMolecule":
        """Create from ChEMBL API response."""
        props = MoleculeProperties.from_dict(data.get("molecule_properties", {}))
        
        return cls(
            molecule_chembl_id=data.get("molecule_chembl_id", ""),
            pref_name=data.get("pref_name", "") or "",
            max_phase=float(data.get("max_phase", 0) or 0),
            molecule_type=data.get("molecule_type", "") or "",
            first_approval=data.get("first_approval"),
            oral=data.get("oral", False) or False,
            parenteral=data.get("parenteral", False) or False,
            topical=data.get("topical", False) or False,
            molecule_synonyms=data.get("molecule_synonyms", []) or [],
            molecule_properties=props,
            molecule_structures=data.get("molecule_structures")
        )
    
    def get_synonyms(self, max_count: int = 10) -> List[str]:
        """Extract synonym names."""
        synonyms = []
        for syn in self.molecule_synonyms[:max_count]:
            if isinstance(syn, dict):
                name = syn.get("molecule_synonym") or syn.get("synonym", "")
            else:
                name = str(syn)
            if name and name.upper() != self.pref_name.upper():
                synonyms.append(name)
        return list(set(synonyms))[:max_count]


@dataclass
class ChEMBLDrugDocument:
    """
    Normalized ChEMBL drug document ready for embedding and Qdrant storage.
    This is the final output format after combining molecule, mechanism, and indication data.
    """
    # Identifiers
    doc_id: str = ""                      # UUID for Qdrant
    chembl_id: str = ""                   # "CHEMBL998"
    drug_name: str = ""                   # "LORATADINE"
    synonyms: List[str] = field(default_factory=list)  # ["Claritin", "Claratyne"]
    
    # Approval Status
    max_phase: int = 0                    # 4 = Approved
    first_approval: Optional[int] = None  # 1993
    approval_status: str = "Unknown"      # "FDA Approved"
    
    # Drug Properties
    molecule_type: str = ""               # "Small molecule"
    oral: bool = False
    parenteral: bool = False
    topical: bool = False
    molecular_weight: Optional[float] = None
    
    # Mechanisms (CRITICAL for gene linking)
    mechanisms: List[MechanismInfo] = field(default_factory=list)
    
    # Indications (for disease matching)
    indications: List[IndicationInfo] = field(default_factory=list)
    
    # Flattened fields for filtering (derived from mechanisms/indications)
    all_gene_symbols: List[str] = field(default_factory=list)      # All target genes
    all_action_types: List[str] = field(default_factory=list)      # All action types
    all_target_names: List[str] = field(default_factory=list)      # All target names
    all_disease_names: List[str] = field(default_factory=list)     # All indication names
    all_mesh_ids: List[str] = field(default_factory=list)          # All MeSH IDs
    all_efo_ids: List[str] = field(default_factory=list)           # All EFO IDs
    
    # Text content for embedding
    text_content: str = ""
    
    # Metadata
    doc_type: str = "chembl_drug"
    data_source: str = "ChEMBL"
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Generate doc_id if not provided."""
        if not self.doc_id:
            self.doc_id = str(uuid.uuid4())
        
        # Derive flattened fields from mechanisms and indications
        self._derive_flattened_fields()
    
    def _derive_flattened_fields(self):
        """Derive flattened fields from nested data."""
        # From mechanisms
        gene_symbols = set()
        action_types = set()
        target_names = set()
        
        for mech in self.mechanisms:
            gene_symbols.update(mech.target_gene_symbols)
            if mech.action_type:
                action_types.add(mech.action_type)
            if mech.target_name:
                target_names.add(mech.target_name)
        
        self.all_gene_symbols = list(gene_symbols)
        self.all_action_types = list(action_types)
        self.all_target_names = list(target_names)
        
        # From indications
        disease_names = set()
        mesh_ids = set()
        efo_ids = set()
        
        for ind in self.indications:
            if ind.mesh_heading:
                disease_names.add(ind.mesh_heading)
            if ind.efo_term:
                disease_names.add(ind.efo_term)
            if ind.mesh_id:
                mesh_ids.add(ind.mesh_id)
            if ind.efo_id:
                efo_ids.add(ind.efo_id)
        
        self.all_disease_names = list(disease_names)
        self.all_mesh_ids = list(mesh_ids)
        self.all_efo_ids = list(efo_ids)
    
    @classmethod
    def from_components(
        cls,
        molecule: ChEMBLMolecule,
        mechanisms: List[MechanismInfo],
        indications: List[IndicationInfo],
        approval_status_map: Dict[float, str] = None
    ) -> "ChEMBLDrugDocument":
        """Create from molecule, mechanisms, and indications."""
        
        # Default approval status mapping
        if approval_status_map is None:
            approval_status_map = {
                4.0: "FDA Approved",
                3.0: "Phase III Clinical Trial",
                2.0: "Phase II Clinical Trial",
                1.0: "Phase I Clinical Trial",
                0.5: "Early Phase I",
                0.0: "Preclinical"
            }
        
        approval_status = approval_status_map.get(
            molecule.max_phase, 
            "Unknown"
        )
        
        molecular_weight = None
        if molecule.molecule_properties:
            molecular_weight = molecule.molecule_properties.molecular_weight
        
        doc = cls(
            chembl_id=molecule.molecule_chembl_id,
            drug_name=molecule.pref_name,
            synonyms=molecule.get_synonyms(),
            max_phase=int(molecule.max_phase),
            first_approval=molecule.first_approval,
            approval_status=approval_status,
            molecule_type=molecule.molecule_type,
            oral=molecule.oral,
            parenteral=molecule.parenteral,
            topical=molecule.topical,
            molecular_weight=molecular_weight,
            mechanisms=mechanisms,
            indications=indications
        )
        
        return doc
    
    def generate_text_content(self) -> str:
        """Generate embedding-optimized text content."""
        parts = []
        
        # Drug identification
        parts.append(
            f"{self.drug_name} ({self.chembl_id}) is a {self.approval_status} {self.molecule_type}."
        )
        
        # Synonyms
        if self.synonyms:
            parts.append(f"Also known as: {', '.join(self.synonyms[:5])}.")
        
        # Mechanisms
        for mech in self.mechanisms[:5]:  # Limit to avoid too long text
            mech_text = f"{self.drug_name} acts as a {mech.action_type} of {mech.target_name}"
            if mech.target_gene_symbols:
                mech_text += f" (gene: {', '.join(mech.target_gene_symbols)})"
            mech_text += f". Mechanism: {mech.mechanism_of_action}."
            parts.append(mech_text)
        
        # Indications
        if self.all_disease_names:
            disease_list = ', '.join(self.all_disease_names[:10])
            parts.append(f"Approved indications include: {disease_list}.")
        
        # Gene summary
        if self.all_gene_symbols:
            parts.append(f"Target genes: {', '.join(self.all_gene_symbols)}.")
        
        text = " ".join(parts)
        
        # Limit length
        if len(text) > 2000:
            text = text[:1997] + "..."
        
        self.text_content = text
        return text
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload format."""
        # Ensure text content is generated
        if not self.text_content:
            self.generate_text_content()
        
        return {
            # Identifiers
            "doc_id": self.doc_id,
            "chembl_id": self.chembl_id,
            "drug_name": self.drug_name,
            "synonyms": self.synonyms,
            
            # Approval
            "max_phase": self.max_phase,
            "approval_status": self.approval_status,
            "first_approval": self.first_approval,
            
            # Properties
            "molecule_type": self.molecule_type,
            "oral": self.oral,
            "parenteral": self.parenteral,
            "topical": self.topical,
            "molecular_weight": self.molecular_weight,
            
            # Mechanisms (flattened for filtering)
            "target_gene_symbols": self.all_gene_symbols,
            "action_types": self.all_action_types,
            "target_names": self.all_target_names,
            
            # Primary mechanism text
            "mechanism_of_action": (
                self.mechanisms[0].mechanism_of_action 
                if self.mechanisms else ""
            ),
            
            # Indications (flattened for filtering)
            "indication_names": self.all_disease_names,
            "indication_mesh_ids": self.all_mesh_ids,
            "indication_efo_ids": self.all_efo_ids,
            
            # Full nested data (for detailed retrieval)
            "mechanisms_full": [m.to_dict() for m in self.mechanisms],
            "indications_full": [i.to_dict() for i in self.indications],
            
            # Text content
            "text_content": self.text_content,
            
            # Metadata
            "doc_type": self.doc_type,
            "data_source": self.data_source,
            "created_at": self.created_at.isoformat()
        }
    
    def has_gene_symbol(self, gene_symbol: str) -> bool:
        """Check if drug targets a specific gene."""
        return gene_symbol.upper() in [g.upper() for g in self.all_gene_symbols]
    
    def has_indication(self, disease_name: str) -> bool:
        """Check if drug has a specific indication."""
        disease_lower = disease_name.lower()
        return any(
            disease_lower in d.lower() 
            for d in self.all_disease_names
        )


@dataclass
class ChEMBLIngestionStats:
    """Statistics from ChEMBL ingestion process."""
    total_molecules_fetched: int = 0
    total_mechanisms_fetched: int = 0
    total_indications_fetched: int = 0
    
    documents_created: int = 0
    documents_ingested: int = 0
    documents_with_mechanisms: int = 0
    documents_with_indications: int = 0
    documents_with_gene_targets: int = 0
    
    unique_gene_symbols: int = 0
    unique_diseases: int = 0
    
    errors: int = 0
    warnings: int = 0
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        """Calculate duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_molecules_fetched": self.total_molecules_fetched,
            "total_mechanisms_fetched": self.total_mechanisms_fetched,
            "total_indications_fetched": self.total_indications_fetched,
            "documents_created": self.documents_created,
            "documents_ingested": self.documents_ingested,
            "documents_with_mechanisms": self.documents_with_mechanisms,
            "documents_with_indications": self.documents_with_indications,
            "documents_with_gene_targets": self.documents_with_gene_targets,
            "unique_gene_symbols": self.unique_gene_symbols,
            "unique_diseases": self.unique_diseases,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_seconds": self.duration_seconds
        }
