"""
Molecule Fetcher
================

Specialized fetcher for ChEMBL approved drug molecules.
Handles pagination, filtering, and data enrichment.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from .chembl_api_client import ChEMBLAPIClient

logger = logging.getLogger(__name__)


@dataclass
class MoleculeFetchResult:
    """Result container for molecule fetch operation."""
    molecules: List[Dict[str, Any]] = field(default_factory=list)
    total_fetched: int = 0
    chembl_ids: Set[str] = field(default_factory=set)
    errors: List[str] = field(default_factory=list)


class MoleculeFetcher:
    """
    Fetcher for ChEMBL approved drug molecules.
    
    Features:
    - Fetch approved drugs (max_phase=4)
    - Filter by molecule type
    - Extract ChEMBL IDs for downstream fetching
    - Cache results for efficiency
    """
    
    def __init__(self, api_client: Optional[ChEMBLAPIClient] = None):
        """
        Initialize molecule fetcher.
        
        Args:
            api_client: ChEMBL API client (creates new if not provided)
        """
        self.api_client = api_client or ChEMBLAPIClient()
    
    def fetch_approved_drugs(
        self,
        limit: Optional[int] = None,
        molecule_types: Optional[List[str]] = None,
        require_name: bool = True,
        use_cache: bool = True
    ) -> MoleculeFetchResult:
        """
        Fetch all approved drugs from ChEMBL.
        
        Args:
            limit: Maximum number of molecules to fetch
            molecule_types: Filter by specific molecule types
            require_name: Only include molecules with a pref_name
            use_cache: Use cached results if available
        
        Returns:
            MoleculeFetchResult with molecules and metadata
        """
        result = MoleculeFetchResult()
        
        try:
            # Fetch raw molecules
            raw_molecules = self.api_client.fetch_approved_molecules(
                max_phase=4,
                limit=limit,
                molecule_types=molecule_types,
                use_cache=use_cache
            )
            
            # Process and filter
            for mol in raw_molecules:
                # Skip molecules without name if required
                if require_name and not mol.get("pref_name"):
                    continue
                
                chembl_id = mol.get("molecule_chembl_id")
                if chembl_id:
                    result.molecules.append(mol)
                    result.chembl_ids.add(chembl_id)
            
            result.total_fetched = len(result.molecules)
            
            logger.info(
                f"Fetched {result.total_fetched} approved drugs "
                f"({len(result.chembl_ids)} unique ChEMBL IDs)"
            )
        
        except Exception as e:
            error_msg = f"Error fetching approved drugs: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        return result
    
    def get_molecule_by_id(self, chembl_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single molecule by ChEMBL ID.
        
        Args:
            chembl_id: ChEMBL ID of the molecule
        
        Returns:
            Molecule dictionary or None
        """
        try:
            molecules = list(
                self.api_client.molecule.filter(molecule_chembl_id=chembl_id)
            )
            if molecules:
                return molecules[0]
            return None
        except Exception as e:
            logger.warning(f"Error fetching molecule {chembl_id}: {e}")
            return None
    
    def get_molecules_by_ids(
        self,
        chembl_ids: List[str],
        batch_size: int = 50
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch multiple molecules by ChEMBL IDs.
        
        Args:
            chembl_ids: List of ChEMBL IDs
            batch_size: Number of molecules per batch request
        
        Returns:
            Dictionary mapping chembl_id to molecule data
        """
        results = {}
        unique_ids = list(set(chembl_ids))
        
        for i in range(0, len(unique_ids), batch_size):
            batch = unique_ids[i:i + batch_size]
            
            try:
                molecules = list(
                    self.api_client.molecule.filter(molecule_chembl_id__in=batch)
                )
                
                for mol in molecules:
                    mol_id = mol.get("molecule_chembl_id")
                    if mol_id:
                        results[mol_id] = mol
            
            except Exception as e:
                logger.warning(f"Error fetching molecule batch: {e}")
                # Fall back to individual fetches
                for mol_id in batch:
                    mol = self.get_molecule_by_id(mol_id)
                    if mol:
                        results[mol_id] = mol
        
        return results
    
    def search_molecules(
        self,
        name_contains: Optional[str] = None,
        max_phase: int = 4,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for molecules by name.
        
        Args:
            name_contains: Search term for molecule name
            max_phase: Maximum clinical phase
            limit: Maximum results to return
        
        Returns:
            List of matching molecules
        """
        try:
            query = self.api_client.molecule.filter(max_phase=max_phase)
            
            if name_contains:
                query = query.filter(pref_name__icontains=name_contains)
            
            results = []
            for mol in query:
                results.append(mol)
                if len(results) >= limit:
                    break
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching molecules: {e}")
            return []
    
    def get_molecule_statistics(
        self,
        molecules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate statistics for a list of molecules.
        
        Args:
            molecules: List of molecule dictionaries
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_count": len(molecules),
            "with_name": 0,
            "molecule_types": {},
            "has_synonyms": 0,
            "has_properties": 0,
            "by_phase": {}
        }
        
        for mol in molecules:
            if mol.get("pref_name"):
                stats["with_name"] += 1
            
            mol_type = mol.get("molecule_type", "Unknown")
            stats["molecule_types"][mol_type] = (
                stats["molecule_types"].get(mol_type, 0) + 1
            )
            
            if mol.get("molecule_synonyms"):
                stats["has_synonyms"] += 1
            
            if mol.get("molecule_properties"):
                stats["has_properties"] += 1
            
            phase = mol.get("max_phase", 0)
            stats["by_phase"][phase] = stats["by_phase"].get(phase, 0) + 1
        
        return stats
