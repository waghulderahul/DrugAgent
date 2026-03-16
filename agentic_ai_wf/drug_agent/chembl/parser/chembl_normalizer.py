"""
ChEMBL Data Normalizer
======================

Normalizes raw ChEMBL API data into standardized document format
ready for embedding and storage in Qdrant.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from ..models.chembl_models import (
    ChEMBLMolecule,
    MechanismInfo,
    IndicationInfo,
    ChEMBLDrugDocument,
)

logger = logging.getLogger(__name__)


# Default mappings
DEFAULT_PHASE_MAPPING = {
    4.0: "FDA Approved",
    4: "FDA Approved",
    3.0: "Phase III Clinical Trial",
    3: "Phase III Clinical Trial",
    2.0: "Phase II Clinical Trial",
    2: "Phase II Clinical Trial",
    1.0: "Phase I Clinical Trial",
    1: "Phase I Clinical Trial",
    0.5: "Early Phase I",
    0.0: "Preclinical",
    0: "Preclinical",
}

DEFAULT_ACTION_TYPE_MAPPING = {
    "INHIBITOR": "Inhibitor",
    "ANTAGONIST": "Antagonist",
    "AGONIST": "Agonist",
    "BLOCKER": "Blocker",
    "MODULATOR": "Modulator",
    "ACTIVATOR": "Activator",
    "OPENER": "Opener",
    "BINDER": "Binder",
    "SUBSTRATE": "Substrate",
    "COFACTOR": "Cofactor",
    "POSITIVE ALLOSTERIC MODULATOR": "Positive Allosteric Modulator",
    "NEGATIVE ALLOSTERIC MODULATOR": "Negative Allosteric Modulator",
    "PARTIAL AGONIST": "Partial Agonist",
    "INVERSE AGONIST": "Inverse Agonist",
}


class ChEMBLNormalizer:
    """
    Normalizer for ChEMBL data.
    
    Transforms raw API responses into standardized ChEMBLDrugDocument objects
    ready for embedding generation and Qdrant storage.
    """
    
    def __init__(
        self,
        phase_mapping: Optional[Dict[float, str]] = None,
        action_type_mapping: Optional[Dict[str, str]] = None,
        max_synonyms: int = 10,
        max_mechanisms: int = 10,
        max_indications: int = 20
    ):
        """
        Initialize normalizer.
        
        Args:
            phase_mapping: Mapping of max_phase to approval status string
            action_type_mapping: Mapping of action types to normalized strings
            max_synonyms: Maximum synonyms to keep per drug
            max_mechanisms: Maximum mechanisms to keep per drug
            max_indications: Maximum indications to keep per drug
        """
        self.phase_mapping = phase_mapping or DEFAULT_PHASE_MAPPING
        self.action_type_mapping = action_type_mapping or DEFAULT_ACTION_TYPE_MAPPING
        self.max_synonyms = max_synonyms
        self.max_mechanisms = max_mechanisms
        self.max_indications = max_indications
    
    def normalize_molecule(self, raw: Dict[str, Any]) -> ChEMBLMolecule:
        """
        Normalize raw molecule data from API.
        
        Args:
            raw: Raw molecule dictionary from ChEMBL API
        
        Returns:
            ChEMBLMolecule object
        """
        return ChEMBLMolecule.from_dict(raw)
    
    def normalize_mechanism(self, raw: Dict[str, Any]) -> MechanismInfo:
        """
        Normalize raw mechanism data from API.
        
        Args:
            raw: Raw mechanism dictionary from ChEMBL API
        
        Returns:
            MechanismInfo object
        """
        # Normalize action type
        action_type = raw.get("action_type", "")
        if action_type:
            action_type = self.action_type_mapping.get(
                action_type.upper(), 
                action_type.title()
            )
            raw["action_type"] = action_type
        
        return MechanismInfo.from_dict(raw)
    
    def normalize_indication(self, raw: Dict[str, Any]) -> IndicationInfo:
        """
        Normalize raw indication data from API.
        
        Args:
            raw: Raw indication dictionary from ChEMBL API
        
        Returns:
            IndicationInfo object
        """
        return IndicationInfo.from_dict(raw)
    
    def create_drug_document(
        self,
        molecule: Dict[str, Any],
        mechanisms: List[Dict[str, Any]],
        indications: List[Dict[str, Any]]
    ) -> ChEMBLDrugDocument:
        """
        Create a complete drug document from molecule, mechanisms, and indications.
        
        Args:
            molecule: Raw molecule dictionary
            mechanisms: List of raw mechanism dictionaries
            indications: List of raw indication dictionaries
        
        Returns:
            ChEMBLDrugDocument ready for embedding
        """
        # Normalize molecule
        norm_molecule = self.normalize_molecule(molecule)
        
        # Normalize mechanisms (limit count)
        norm_mechanisms = []
        for mech in mechanisms[:self.max_mechanisms]:
            norm_mech = self.normalize_mechanism(mech)
            norm_mechanisms.append(norm_mech)
        
        # Normalize indications (limit count)
        norm_indications = []
        for ind in indications[:self.max_indications]:
            norm_ind = self.normalize_indication(ind)
            norm_indications.append(norm_ind)
        
        # Create document
        doc = ChEMBLDrugDocument.from_components(
            molecule=norm_molecule,
            mechanisms=norm_mechanisms,
            indications=norm_indications,
            approval_status_map=self.phase_mapping
        )
        
        # Limit synonyms
        doc.synonyms = doc.synonyms[:self.max_synonyms]
        
        # Generate text content for embedding
        doc.generate_text_content()
        
        return doc
    
    def normalize_batch(
        self,
        molecules: List[Dict[str, Any]],
        mechanisms_by_molecule: Dict[str, List[Dict[str, Any]]],
        indications_by_molecule: Dict[str, List[Dict[str, Any]]],
        skip_on_error: bool = True
    ) -> Tuple[List[ChEMBLDrugDocument], List[str]]:
        """
        Normalize a batch of molecules with their mechanisms and indications.
        
        Args:
            molecules: List of raw molecule dictionaries
            mechanisms_by_molecule: Mechanisms indexed by molecule ChEMBL ID
            indications_by_molecule: Indications indexed by molecule ChEMBL ID
            skip_on_error: Skip molecules that fail normalization
        
        Returns:
            Tuple of (list of documents, list of error messages)
        """
        documents = []
        errors = []
        
        for mol in molecules:
            chembl_id = mol.get("molecule_chembl_id")
            if not chembl_id:
                continue
            
            try:
                # Get mechanisms and indications for this molecule
                mol_mechanisms = mechanisms_by_molecule.get(chembl_id, [])
                mol_indications = indications_by_molecule.get(chembl_id, [])
                
                # Create document
                doc = self.create_drug_document(
                    molecule=mol,
                    mechanisms=mol_mechanisms,
                    indications=mol_indications
                )
                
                documents.append(doc)
            
            except Exception as e:
                error_msg = f"Error normalizing {chembl_id}: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
                
                if not skip_on_error:
                    raise
        
        logger.info(
            f"Normalized {len(documents)} documents "
            f"({len(errors)} errors)"
        )
        
        return documents, errors
    
    def validate_document(
        self,
        doc: ChEMBLDrugDocument,
        required_fields: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate a drug document.
        
        Args:
            doc: Document to validate
            required_fields: List of required fields
        
        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        if required_fields is None:
            required_fields = ["chembl_id", "drug_name"]
        
        errors = []
        
        # Check required fields
        for field in required_fields:
            value = getattr(doc, field, None)
            if not value:
                errors.append(f"Missing required field: {field}")
        
        # Check text content
        if not doc.text_content:
            errors.append("Missing text_content")
        
        # Warn about missing useful data (not errors)
        warnings = []
        if not doc.mechanisms:
            warnings.append("No mechanisms of action")
        if not doc.all_gene_symbols:
            warnings.append("No target gene symbols")
        if not doc.indications:
            warnings.append("No indications")
        
        for warning in warnings:
            logger.debug(f"Document {doc.chembl_id}: {warning}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_document_statistics(
        self,
        documents: List[ChEMBLDrugDocument]
    ) -> Dict[str, Any]:
        """
        Calculate statistics for a list of documents.
        
        Args:
            documents: List of ChEMBLDrugDocument objects
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_documents": len(documents),
            "with_mechanisms": 0,
            "with_indications": 0,
            "with_gene_targets": 0,
            "unique_gene_symbols": set(),
            "unique_diseases": set(),
            "molecule_types": defaultdict(int),
            "approval_statuses": defaultdict(int),
            "action_types": defaultdict(int),
            "avg_mechanisms_per_drug": 0,
            "avg_indications_per_drug": 0,
            "avg_text_length": 0
        }
        
        total_mechanisms = 0
        total_indications = 0
        total_text_length = 0
        
        for doc in documents:
            # Count documents with data
            if doc.mechanisms:
                stats["with_mechanisms"] += 1
                total_mechanisms += len(doc.mechanisms)
            
            if doc.indications:
                stats["with_indications"] += 1
                total_indications += len(doc.indications)
            
            if doc.all_gene_symbols:
                stats["with_gene_targets"] += 1
                stats["unique_gene_symbols"].update(doc.all_gene_symbols)
            
            # Collect unique diseases
            stats["unique_diseases"].update(doc.all_disease_names)
            
            # Count molecule types
            if doc.molecule_type:
                stats["molecule_types"][doc.molecule_type] += 1
            
            # Count approval statuses
            if doc.approval_status:
                stats["approval_statuses"][doc.approval_status] += 1
            
            # Count action types
            for action_type in doc.all_action_types:
                stats["action_types"][action_type] += 1
            
            # Text length
            total_text_length += len(doc.text_content)
        
        # Calculate averages
        if len(documents) > 0:
            stats["avg_mechanisms_per_drug"] = round(
                total_mechanisms / len(documents), 2
            )
            stats["avg_indications_per_drug"] = round(
                total_indications / len(documents), 2
            )
            stats["avg_text_length"] = round(
                total_text_length / len(documents), 0
            )
        
        # Convert sets to counts for JSON serialization
        stats["unique_gene_symbols"] = len(stats["unique_gene_symbols"])
        stats["unique_diseases"] = len(stats["unique_diseases"])
        stats["molecule_types"] = dict(stats["molecule_types"])
        stats["approval_statuses"] = dict(stats["approval_statuses"])
        stats["action_types"] = dict(stats["action_types"])
        
        return stats
