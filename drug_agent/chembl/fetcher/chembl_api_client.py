"""
ChEMBL API Client
=================

Main API wrapper for ChEMBL web services with retry logic, caching, and error handling.
Uses the official chembl_webresource_client library.

Usage:
------
    from chembl.fetcher import ChEMBLAPIClient
    
    client = ChEMBLAPIClient()
    molecules = client.fetch_approved_molecules(limit=100)
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed: {e}")
            raise last_exception
        return wrapper
    return decorator


class ChEMBLAPIClient:
    """
    ChEMBL API client with caching and retry logic.
    
    This client wraps the chembl_webresource_client library and adds:
    - Automatic retry with exponential backoff
    - Response caching to disk
    - Progress tracking for large fetches
    - Rate limiting support
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        cache_enabled: bool = True,
        cache_expiry_days: int = 30,
        max_retries: int = 3,
        timeout: int = 60
    ):
        """
        Initialize ChEMBL API client.
        
        Args:
            cache_dir: Directory for caching API responses
            cache_enabled: Whether to enable caching
            cache_expiry_days: Days until cache expires (0 = never)
            max_retries: Maximum retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        self.cache_enabled = cache_enabled
        self.cache_expiry_days = cache_expiry_days
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Set cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent / "cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API clients (lazy loaded)
        self._molecule_client = None
        self._mechanism_client = None
        self._indication_client = None
        self._target_client = None
        self._activity_client = None
        
        logger.info(f"ChEMBL API client initialized. Cache: {self.cache_dir}")
    
    def _init_clients(self):
        """Lazy initialize ChEMBL API clients."""
        if self._molecule_client is None:
            try:
                from chembl_webresource_client.new_client import new_client
                
                self._molecule_client = new_client.molecule
                self._mechanism_client = new_client.mechanism
                self._indication_client = new_client.drug_indication
                self._target_client = new_client.target
                self._activity_client = new_client.activity
                
                logger.info("ChEMBL API clients initialized successfully")
            except ImportError as e:
                logger.error(
                    "chembl_webresource_client not installed. "
                    "Run: pip install chembl_webresource_client"
                )
                raise ImportError(
                    "Please install chembl_webresource_client: "
                    "pip install chembl_webresource_client"
                ) from e
    
    @property
    def molecule(self):
        """Get molecule API client."""
        self._init_clients()
        return self._molecule_client
    
    @property
    def mechanism(self):
        """Get mechanism API client."""
        self._init_clients()
        return self._mechanism_client
    
    @property
    def indication(self):
        """Get drug indication API client."""
        self._init_clients()
        return self._indication_client
    
    @property
    def target(self):
        """Get target API client."""
        self._init_clients()
        return self._target_client
    
    @property
    def activity(self):
        """Get activity API client."""
        self._init_clients()
        return self._activity_client
    
    # ==================== Caching Methods ====================
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a key."""
        # Sanitize cache key for filename
        safe_key = "".join(c if c.isalnum() or c in "._-" else "_" for c in cache_key)
        return self.cache_dir / f"{safe_key}.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is not expired."""
        if not cache_path.exists():
            return False
        
        if self.cache_expiry_days == 0:
            return True  # Never expires
        
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = modified_time + timedelta(days=self.cache_expiry_days)
        
        return datetime.now() < expiry_time
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache if valid."""
        if not self.cache_enabled:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.debug(f"Loaded from cache: {cache_key}")
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Any):
        """Save data to cache."""
        if not self.cache_enabled:
            return
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Saved to cache: {cache_key}")
        except IOError as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def clear_cache(self, cache_key: Optional[str] = None):
        """Clear cache files."""
        if cache_key:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cleared cache: {cache_key}")
        else:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cleared all cache files")
    
    # ==================== Fetch Methods ====================
    
    @retry_with_backoff(max_retries=3)
    def fetch_approved_molecules(
        self,
        max_phase: int = 4,
        limit: Optional[int] = None,
        molecule_types: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch approved drug molecules from ChEMBL.
        
        Args:
            max_phase: Maximum clinical phase (4 = approved)
            limit: Maximum number of molecules to fetch (None = all)
            molecule_types: Filter by molecule type
            use_cache: Whether to use cached data if available
        
        Returns:
            List of molecule dictionaries
        """
        cache_key = f"molecules_phase{max_phase}"
        if molecule_types:
            cache_key += f"_types_{'_'.join(molecule_types)}"
        
        # Try cache first
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached:
                logger.info(f"Using cached molecules: {len(cached)} items")
                if limit:
                    return cached[:limit]
                return cached
        
        logger.info(f"Fetching molecules with max_phase={max_phase}...")
        
        # Build query
        query = self.molecule.filter(max_phase=max_phase)
        
        # Convert to list (this fetches all pages)
        molecules = []
        try:
            # The client returns a QuerySet-like object
            # Converting to list fetches all data
            for mol in query:
                molecules.append(mol)
                
                if limit and len(molecules) >= limit:
                    break
                
                # Progress logging
                if len(molecules) % 500 == 0:
                    logger.info(f"Fetched {len(molecules)} molecules...")
        
        except Exception as e:
            logger.error(f"Error fetching molecules: {e}")
            raise
        
        logger.info(f"Fetched {len(molecules)} molecules total")
        
        # Filter by molecule type if specified
        if molecule_types:
            molecules = [
                m for m in molecules 
                if m.get("molecule_type") in molecule_types
            ]
            logger.info(f"Filtered to {len(molecules)} molecules by type")
        
        # Cache the results
        self._save_to_cache(cache_key, molecules)
        
        return molecules
    
    @retry_with_backoff(max_retries=3)
    def fetch_mechanisms_for_molecule(
        self,
        molecule_chembl_id: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch mechanisms of action for a specific molecule.
        
        Args:
            molecule_chembl_id: ChEMBL ID of the molecule
        
        Returns:
            List of mechanism dictionaries
        """
        try:
            mechanisms = list(
                self.mechanism.filter(molecule_chembl_id=molecule_chembl_id)
            )
            return mechanisms
        except Exception as e:
            logger.warning(f"Error fetching mechanisms for {molecule_chembl_id}: {e}")
            return []
    
    @retry_with_backoff(max_retries=3)
    def fetch_all_mechanisms(
        self,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch all mechanisms of action from ChEMBL.
        
        Args:
            use_cache: Whether to use cached data
        
        Returns:
            List of all mechanism dictionaries
        """
        cache_key = "mechanisms_all"
        
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached:
                logger.info(f"Using cached mechanisms: {len(cached)} items")
                return cached
        
        logger.info("Fetching all mechanisms...")
        
        mechanisms = []
        try:
            for mech in self.mechanism.all():
                mechanisms.append(mech)
                
                if len(mechanisms) % 1000 == 0:
                    logger.info(f"Fetched {len(mechanisms)} mechanisms...")
        
        except Exception as e:
            logger.error(f"Error fetching mechanisms: {e}")
            raise
        
        logger.info(f"Fetched {len(mechanisms)} mechanisms total")
        self._save_to_cache(cache_key, mechanisms)
        
        return mechanisms
    
    @retry_with_backoff(max_retries=3)
    def fetch_indications_for_molecule(
        self,
        molecule_chembl_id: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch drug indications for a specific molecule.
        
        Args:
            molecule_chembl_id: ChEMBL ID of the molecule
        
        Returns:
            List of indication dictionaries
        """
        try:
            indications = list(
                self.indication.filter(molecule_chembl_id=molecule_chembl_id)
            )
            return indications
        except Exception as e:
            logger.warning(f"Error fetching indications for {molecule_chembl_id}: {e}")
            return []
    
    @retry_with_backoff(max_retries=3)
    def fetch_all_indications(
        self,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch all drug indications from ChEMBL.
        
        Args:
            use_cache: Whether to use cached data
        
        Returns:
            List of all indication dictionaries
        """
        cache_key = "indications_all"
        
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached:
                logger.info(f"Using cached indications: {len(cached)} items")
                return cached
        
        logger.info("Fetching all indications...")
        
        indications = []
        try:
            for ind in self.indication.all():
                indications.append(ind)
                
                if len(indications) % 1000 == 0:
                    logger.info(f"Fetched {len(indications)} indications...")
        
        except Exception as e:
            logger.error(f"Error fetching indications: {e}")
            raise
        
        logger.info(f"Fetched {len(indications)} indications total")
        self._save_to_cache(cache_key, indications)
        
        return indications
    
    @retry_with_backoff(max_retries=3)
    def fetch_target(self, target_chembl_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch target details by ChEMBL ID.
        
        Args:
            target_chembl_id: ChEMBL ID of the target
        
        Returns:
            Target dictionary or None
        """
        try:
            targets = list(
                self.target.filter(target_chembl_id=target_chembl_id)
            )
            if targets:
                return targets[0]
            return None
        except Exception as e:
            logger.warning(f"Error fetching target {target_chembl_id}: {e}")
            return None
    
    def fetch_targets_batch(
        self,
        target_chembl_ids: List[str],
        batch_size: int = 50
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch multiple targets in batches.
        
        Args:
            target_chembl_ids: List of target ChEMBL IDs
            batch_size: Number of targets per batch
        
        Returns:
            Dictionary mapping target_chembl_id to target data
        """
        results = {}
        unique_ids = list(set(target_chembl_ids))
        
        logger.info(f"Fetching {len(unique_ids)} unique targets...")
        
        for i in range(0, len(unique_ids), batch_size):
            batch = unique_ids[i:i + batch_size]
            
            try:
                # Use __in filter for batch fetching
                targets = list(
                    self.target.filter(target_chembl_id__in=batch)
                )
                
                for target in targets:
                    target_id = target.get("target_chembl_id")
                    if target_id:
                        results[target_id] = target
                
            except Exception as e:
                logger.warning(f"Error fetching target batch: {e}")
                # Fall back to individual fetches
                for target_id in batch:
                    target = self.fetch_target(target_id)
                    if target:
                        results[target_id] = target
            
            if (i + batch_size) % 200 == 0:
                logger.info(f"Fetched {min(i + batch_size, len(unique_ids))} targets...")
        
        logger.info(f"Fetched {len(results)} targets total")
        return results
    
    # ==================== Utility Methods ====================
    
    def test_connection(self) -> bool:
        """Test connection to ChEMBL API."""
        try:
            self._init_clients()
            # Try a simple query
            result = list(self.molecule.filter(molecule_chembl_id="CHEMBL25")[:1])
            if result:
                logger.info("ChEMBL API connection successful")
                return True
            return False
        except Exception as e:
            logger.error(f"ChEMBL API connection failed: {e}")
            return False
    
    def get_molecule_count(self, max_phase: int = 4) -> int:
        """Get count of molecules with specified max_phase."""
        try:
            # This is an estimate - the client doesn't have a direct count method
            query = self.molecule.filter(max_phase=max_phase)
            # Get first page to estimate
            count = 0
            for _ in query:
                count += 1
                if count >= 10000:  # Safety limit
                    break
            return count
        except Exception as e:
            logger.error(f"Error getting molecule count: {e}")
            return 0
