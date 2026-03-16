"""
Indication Fetcher
==================

Fetcher for ChEMBL drug indication (disease) data.
Links drugs to diseases/conditions they are approved for.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

from .chembl_api_client import ChEMBLAPIClient

logger = logging.getLogger(__name__)


@dataclass
class IndicationFetchResult:
    """Result container for indication fetch operation."""
    # Indications grouped by molecule
    indications_by_molecule: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    total_fetched: int = 0
    unique_mesh_ids: Set[str] = field(default_factory=set)
    unique_efo_ids: Set[str] = field(default_factory=set)
    unique_disease_names: Set[str] = field(default_factory=set)
    errors: List[str] = field(default_factory=list)


class IndicationFetcher:
    """
    Fetcher for ChEMBL drug indication data.
    
    Features:
    - Fetch indications for specific molecules
    - Batch fetch all indications
    - Extract disease names (MeSH and EFO terms)
    - Group indications by molecule
    """
    
    def __init__(self, api_client: Optional[ChEMBLAPIClient] = None):
        """
        Initialize indication fetcher.
        
        Args:
            api_client: ChEMBL API client
        """
        self.api_client = api_client or ChEMBLAPIClient()
    
    def fetch_indications_for_molecules(
        self,
        chembl_ids: List[str],
        use_cache: bool = True
    ) -> IndicationFetchResult:
        """
        Fetch indications for a list of molecules.
        
        Args:
            chembl_ids: List of molecule ChEMBL IDs
            use_cache: Use cached data if available
        
        Returns:
            IndicationFetchResult with indications grouped by molecule
        """
        result = IndicationFetchResult()
        
        logger.info(f"Fetching indications for {len(chembl_ids)} molecules...")
        
        # Fetch all indications at once (more efficient)
        all_indications = self._fetch_all_indications_cached(use_cache)
        
        # Index indications by molecule
        ind_index = defaultdict(list)
        for ind in all_indications:
            mol_id = ind.get("molecule_chembl_id")
            if mol_id:
                ind_index[mol_id].append(ind)
        
        # Collect indications for requested molecules
        for chembl_id in chembl_ids:
            if chembl_id in ind_index:
                for ind in ind_index[chembl_id]:
                    result.indications_by_molecule[chembl_id].append(ind)
                    result.total_fetched += 1
                    
                    # Track unique identifiers
                    mesh_id = ind.get("mesh_id")
                    if mesh_id:
                        result.unique_mesh_ids.add(mesh_id)
                    
                    efo_id = ind.get("efo_id")
                    if efo_id:
                        result.unique_efo_ids.add(efo_id)
                    
                    # Track disease names
                    mesh_heading = ind.get("mesh_heading")
                    if mesh_heading:
                        result.unique_disease_names.add(mesh_heading)
                    
                    efo_term = ind.get("efo_term")
                    if efo_term:
                        result.unique_disease_names.add(efo_term)
        
        logger.info(
            f"Found {result.total_fetched} indications for "
            f"{len(result.indications_by_molecule)} molecules"
        )
        logger.info(f"Found {len(result.unique_disease_names)} unique disease names")
        
        return result
    
    def _fetch_all_indications_cached(
        self,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Fetch all indications with caching."""
        return self.api_client.fetch_all_indications(use_cache=use_cache)
    
    def fetch_indications_for_molecule(
        self,
        chembl_id: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch indications for a single molecule.
        
        Args:
            chembl_id: ChEMBL ID of the molecule
        
        Returns:
            List of indication dictionaries
        """
        return self.api_client.fetch_indications_for_molecule(chembl_id)
    
    def get_disease_names_for_molecule(
        self,
        chembl_id: str,
        indications: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Extract all disease names associated with a molecule's indications.
        
        Args:
            chembl_id: ChEMBL ID of the molecule
            indications: Pre-fetched indications (fetches if not provided)
        
        Returns:
            List of disease names (MeSH headings and EFO terms)
        """
        if indications is None:
            indications = self.fetch_indications_for_molecule(chembl_id)
        
        disease_names = set()
        
        for ind in indications:
            mesh_heading = ind.get("mesh_heading")
            if mesh_heading:
                disease_names.add(mesh_heading)
            
            efo_term = ind.get("efo_term")
            if efo_term:
                disease_names.add(efo_term)
        
        return list(disease_names)
    
    def search_by_disease(
        self,
        disease_term: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for molecules indicated for a disease.
        
        Args:
            disease_term: Disease name to search for
            limit: Maximum results to return
        
        Returns:
            List of indication dictionaries with molecule info
        """
        try:
            # Search by MeSH heading
            results = list(
                self.api_client.indication.filter(
                    mesh_heading__icontains=disease_term
                )
            )[:limit]
            
            # Also search by EFO term
            efo_results = list(
                self.api_client.indication.filter(
                    efo_term__icontains=disease_term
                )
            )[:limit]
            
            # Combine and deduplicate
            seen_ids = set()
            combined = []
            
            for ind in results + efo_results:
                key = (
                    ind.get("molecule_chembl_id"), 
                    ind.get("mesh_id", ""), 
                    ind.get("efo_id", "")
                )
                if key not in seen_ids:
                    seen_ids.add(key)
                    combined.append(ind)
            
            return combined[:limit]
        
        except Exception as e:
            logger.error(f"Error searching indications by disease: {e}")
            return []
    
    def get_molecules_for_disease(
        self,
        disease_term: str,
        min_phase: int = 3
    ) -> List[str]:
        """
        Get ChEMBL IDs of molecules indicated for a disease.
        
        Args:
            disease_term: Disease name to search for
            min_phase: Minimum clinical phase for the indication
        
        Returns:
            List of molecule ChEMBL IDs
        """
        indications = self.search_by_disease(disease_term, limit=500)
        
        molecule_ids = set()
        for ind in indications:
            phase = ind.get("max_phase_for_ind", 0) or 0
            if phase >= min_phase:
                mol_id = ind.get("molecule_chembl_id")
                if mol_id:
                    molecule_ids.add(mol_id)
        
        return list(molecule_ids)
    
    def get_indication_statistics(
        self,
        indications_by_molecule: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Calculate statistics for indications.
        
        Args:
            indications_by_molecule: Indications grouped by molecule
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_molecules": len(indications_by_molecule),
            "total_indications": 0,
            "molecules_with_indications": 0,
            "unique_mesh_ids": set(),
            "unique_efo_ids": set(),
            "unique_diseases": set(),
            "phase_distribution": {}
        }
        
        for mol_id, indications in indications_by_molecule.items():
            if indications:
                stats["molecules_with_indications"] += 1
                
                for ind in indications:
                    stats["total_indications"] += 1
                    
                    mesh_id = ind.get("mesh_id")
                    if mesh_id:
                        stats["unique_mesh_ids"].add(mesh_id)
                    
                    efo_id = ind.get("efo_id")
                    if efo_id:
                        stats["unique_efo_ids"].add(efo_id)
                    
                    mesh_heading = ind.get("mesh_heading")
                    if mesh_heading:
                        stats["unique_diseases"].add(mesh_heading)
                    
                    phase = ind.get("max_phase_for_ind", 0)
                    stats["phase_distribution"][phase] = (
                        stats["phase_distribution"].get(phase, 0) + 1
                    )
        
        # Convert sets to counts for JSON serialization
        stats["unique_mesh_ids"] = len(stats["unique_mesh_ids"])
        stats["unique_efo_ids"] = len(stats["unique_efo_ids"])
        stats["unique_diseases"] = len(stats["unique_diseases"])
        
        return stats
