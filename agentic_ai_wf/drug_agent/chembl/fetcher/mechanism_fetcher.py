"""
Mechanism Fetcher
=================

Fetcher for ChEMBL mechanism of action data.
Extracts target gene symbols for drug-gene linking.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

from .chembl_api_client import ChEMBLAPIClient

logger = logging.getLogger(__name__)


@dataclass
class MechanismFetchResult:
    """Result container for mechanism fetch operation."""
    # Mechanisms grouped by molecule
    mechanisms_by_molecule: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    total_fetched: int = 0
    unique_targets: Set[str] = field(default_factory=set)
    unique_gene_symbols: Set[str] = field(default_factory=set)
    errors: List[str] = field(default_factory=list)


class MechanismFetcher:
    """
    Fetcher for ChEMBL mechanism of action data.
    
    Features:
    - Fetch mechanisms for specific molecules
    - Batch fetch all mechanisms
    - Extract target gene symbols
    - Group mechanisms by molecule
    """
    
    def __init__(self, api_client: Optional[ChEMBLAPIClient] = None):
        """
        Initialize mechanism fetcher.
        
        Args:
            api_client: ChEMBL API client
        """
        self.api_client = api_client or ChEMBLAPIClient()
        self._target_cache: Dict[str, Dict[str, Any]] = {}
    
    def fetch_mechanisms_for_molecules(
        self,
        chembl_ids: List[str],
        enrich_with_targets: bool = True,
        use_cache: bool = True
    ) -> MechanismFetchResult:
        """
        Fetch mechanisms for a list of molecules.
        
        Args:
            chembl_ids: List of molecule ChEMBL IDs
            enrich_with_targets: Fetch additional target data for gene symbols
            use_cache: Use cached data if available
        
        Returns:
            MechanismFetchResult with mechanisms grouped by molecule
        """
        result = MechanismFetchResult()
        
        logger.info(f"Fetching mechanisms for {len(chembl_ids)} molecules...")
        
        # First, try to get all mechanisms at once (more efficient)
        all_mechanisms = self._fetch_all_mechanisms_cached(use_cache)
        
        # Index mechanisms by molecule
        mech_index = defaultdict(list)
        for mech in all_mechanisms:
            mol_id = mech.get("molecule_chembl_id")
            if mol_id:
                mech_index[mol_id].append(mech)
        
        # Collect mechanisms for requested molecules
        target_ids_to_fetch = set()
        
        for chembl_id in chembl_ids:
            if chembl_id in mech_index:
                for mech in mech_index[chembl_id]:
                    result.mechanisms_by_molecule[chembl_id].append(mech)
                    result.total_fetched += 1
                    
                    # Track unique targets
                    target_id = mech.get("target_chembl_id")
                    if target_id:
                        result.unique_targets.add(target_id)
                        target_ids_to_fetch.add(target_id)
                    
                    # Extract gene symbols from target_components if available
                    for comp in mech.get("target_components", []):
                        gene_symbol = comp.get("gene_symbol")
                        if gene_symbol:
                            result.unique_gene_symbols.add(gene_symbol)
        
        logger.info(
            f"Found {result.total_fetched} mechanisms for "
            f"{len(result.mechanisms_by_molecule)} molecules"
        )
        
        # Enrich with target data to get gene symbols
        if enrich_with_targets and target_ids_to_fetch:
            self._enrich_with_target_data(
                result, 
                list(target_ids_to_fetch)
            )
        
        logger.info(f"Found {len(result.unique_gene_symbols)} unique gene symbols")
        
        return result
    
    def _fetch_all_mechanisms_cached(
        self,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Fetch all mechanisms with caching."""
        return self.api_client.fetch_all_mechanisms(use_cache=use_cache)
    
    def _enrich_with_target_data(
        self,
        result: MechanismFetchResult,
        target_ids: List[str]
    ):
        """
        Enrich mechanisms with target data to extract gene symbols.
        
        Args:
            result: MechanismFetchResult to enrich
            target_ids: List of target ChEMBL IDs to fetch
        """
        logger.info(f"Enriching with target data for {len(target_ids)} targets...")
        
        # Fetch targets not in cache
        uncached_ids = [
            tid for tid in target_ids 
            if tid not in self._target_cache
        ]
        
        if uncached_ids:
            targets = self.api_client.fetch_targets_batch(uncached_ids)
            self._target_cache.update(targets)
        
        # Extract gene symbols from targets
        for mol_id, mechanisms in result.mechanisms_by_molecule.items():
            for mech in mechanisms:
                target_id = mech.get("target_chembl_id")
                if target_id and target_id in self._target_cache:
                    target = self._target_cache[target_id]
                    
                    # Extract gene symbols from target components
                    for comp in target.get("target_components", []):
                        # Try direct gene_symbol field first
                        gene_symbol = comp.get("gene_symbol")
                        if gene_symbol:
                            result.unique_gene_symbols.add(gene_symbol)
                            if "target_gene_symbols" not in mech:
                                mech["target_gene_symbols"] = []
                            if gene_symbol not in mech["target_gene_symbols"]:
                                mech["target_gene_symbols"].append(gene_symbol)
                        
                        # Also look in target_component_synonyms for GENE_SYMBOL type
                        for syn in comp.get("target_component_synonyms", []):
                            if syn.get("syn_type") == "GENE_SYMBOL":
                                gene_symbol = syn.get("component_synonym")
                                if gene_symbol:
                                    result.unique_gene_symbols.add(gene_symbol)
                                    if "target_gene_symbols" not in mech:
                                        mech["target_gene_symbols"] = []
                                    if gene_symbol not in mech["target_gene_symbols"]:
                                        mech["target_gene_symbols"].append(gene_symbol)
    
    def fetch_mechanism_for_molecule(
        self,
        chembl_id: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch mechanisms for a single molecule.
        
        Args:
            chembl_id: ChEMBL ID of the molecule
        
        Returns:
            List of mechanism dictionaries
        """
        return self.api_client.fetch_mechanisms_for_molecule(chembl_id)
    
    def get_gene_symbols_for_molecule(
        self,
        chembl_id: str,
        mechanisms: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Extract all gene symbols associated with a molecule's mechanisms.
        
        Args:
            chembl_id: ChEMBL ID of the molecule
            mechanisms: Pre-fetched mechanisms (fetches if not provided)
        
        Returns:
            List of gene symbols
        """
        if mechanisms is None:
            mechanisms = self.fetch_mechanism_for_molecule(chembl_id)
        
        gene_symbols = set()
        target_ids = []
        
        # First try to get from target_gene_symbols already added
        for mech in mechanisms:
            # Check existing target_gene_symbols
            for gene in mech.get("target_gene_symbols", []):
                gene_symbols.add(gene)
            
            # Check target_components for gene_symbol field
            for comp in mech.get("target_components", []):
                gene = comp.get("gene_symbol")
                if gene:
                    gene_symbols.add(gene)
                # Also check synonyms
                for syn in comp.get("target_component_synonyms", []):
                    if syn.get("syn_type") == "GENE_SYMBOL":
                        gene = syn.get("component_synonym")
                        if gene:
                            gene_symbols.add(gene)
            
            # Collect target IDs for additional lookup
            target_id = mech.get("target_chembl_id")
            if target_id:
                target_ids.append(target_id)
        
        # If we didn't find gene symbols, try fetching target details
        if not gene_symbols and target_ids:
            for target_id in target_ids:
                if target_id not in self._target_cache:
                    target = self.api_client.fetch_target(target_id)
                    if target:
                        self._target_cache[target_id] = target
                
                if target_id in self._target_cache:
                    target = self._target_cache[target_id]
                    for comp in target.get("target_components", []):
                        # Try direct gene_symbol
                        gene = comp.get("gene_symbol")
                        if gene:
                            gene_symbols.add(gene)
                        # Also check synonyms for GENE_SYMBOL type
                        for syn in comp.get("target_component_synonyms", []):
                            if syn.get("syn_type") == "GENE_SYMBOL":
                                gene = syn.get("component_synonym")
                                if gene:
                                    gene_symbols.add(gene)
        
        return list(gene_symbols)
    
    def get_mechanism_statistics(
        self,
        mechanisms_by_molecule: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Calculate statistics for mechanisms.
        
        Args:
            mechanisms_by_molecule: Mechanisms grouped by molecule
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_molecules": len(mechanisms_by_molecule),
            "total_mechanisms": 0,
            "action_types": {},
            "target_types": {},
            "molecules_with_mechanisms": 0,
            "molecules_with_gene_targets": 0
        }
        
        for mol_id, mechanisms in mechanisms_by_molecule.items():
            if mechanisms:
                stats["molecules_with_mechanisms"] += 1
                
                has_gene = False
                for mech in mechanisms:
                    stats["total_mechanisms"] += 1
                    
                    action_type = mech.get("action_type", "Unknown")
                    stats["action_types"][action_type] = (
                        stats["action_types"].get(action_type, 0) + 1
                    )
                    
                    target_type = mech.get("target_type", "Unknown")
                    stats["target_types"][target_type] = (
                        stats["target_types"].get(target_type, 0) + 1
                    )
                    
                    if mech.get("target_gene_symbols") or mech.get("target_components"):
                        has_gene = True
                
                if has_gene:
                    stats["molecules_with_gene_targets"] += 1
        
        return stats
