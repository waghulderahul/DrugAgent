"""
ChEMBL Ingestion Pipeline
=========================

Main orchestrator for fetching, normalizing, embedding, and storing
ChEMBL drug data into Qdrant vector database.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

from ..config.settings import load_chembl_config, ChEMBLConfig, get_cache_dir
from ..models.chembl_models import ChEMBLDrugDocument, ChEMBLIngestionStats
from ..fetcher import (
    ChEMBLAPIClient,
    MoleculeFetcher,
    MechanismFetcher,
    IndicationFetcher,
)
from ..parser import ChEMBLNormalizer

logger = logging.getLogger(__name__)


class ChEMBLIngestion:
    """
    Main orchestrator for ChEMBL data ingestion.
    
    Pipeline:
    1. Fetch approved molecules from ChEMBL API
    2. Fetch mechanisms of action
    3. Fetch drug indications
    4. Normalize data into documents
    5. Generate embeddings with PubMedBERT
    6. Store in Qdrant (ChEMBL_drugs collection)
    
    Features:
    - Checkpoint/resume support
    - Progress tracking
    - Batch processing
    - Error handling with skip option
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_username: Optional[str] = None,
        qdrant_password: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            config_path: Path to chembl_config.yaml
            qdrant_url: Override Qdrant URL from env
            qdrant_api_key: Override Qdrant API key from env
            qdrant_username: Qdrant username for Basic Auth
            qdrant_password: Qdrant password for Basic Auth
            collection_name: Override collection name
        """
        # Load environment variables
        env_path = Path(__file__).parent.parent.parent / '.env'
        load_dotenv(env_path)
        
        # Load configuration
        self.config = load_chembl_config(config_path)
        
        # Qdrant connection settings
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.qdrant_username = qdrant_username or os.getenv("QDRANT_USERNAME")
        self.qdrant_password = qdrant_password or os.getenv("QDRANT_PASSWORD")
        self.collection_name = (
            collection_name or 
            os.getenv("CHEMBL_COLLECTION") or 
            self.config.qdrant.collection_name
        )
        
        # Initialize components
        self.api_client = None
        self.molecule_fetcher = None
        self.mechanism_fetcher = None
        self.indication_fetcher = None
        self.normalizer = None
        self.embedder = None
        self.qdrant_client = None
        
        # Stats tracking
        self.stats = ChEMBLIngestionStats()
        
        # Cache directory
        self.cache_dir = get_cache_dir(self.config)
        
        logger.info(f"ChEMBL Ingestion initialized")
        logger.info(f"Collection: {self.collection_name}")
        logger.info(f"Cache dir: {self.cache_dir}")
    
    def _init_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")
        
        # API client and fetchers
        self.api_client = ChEMBLAPIClient(
            cache_dir=str(self.cache_dir),
            cache_enabled=self.config.cache.enabled,
            cache_expiry_days=self.config.cache.expiry_days,
            max_retries=self.config.chembl_api.max_retries
        )
        
        self.molecule_fetcher = MoleculeFetcher(self.api_client)
        self.mechanism_fetcher = MechanismFetcher(self.api_client)
        self.indication_fetcher = IndicationFetcher(self.api_client)
        
        # Normalizer
        self.normalizer = ChEMBLNormalizer(
            max_synonyms=self.config.normalization.max_synonyms if hasattr(self.config, 'normalization') else 10,
            max_mechanisms=self.config.normalization.max_mechanisms if hasattr(self.config, 'normalization') else 10,
            max_indications=self.config.normalization.max_indications if hasattr(self.config, 'normalization') else 20
        )
        
        logger.info("Pipeline components initialized")
    
    def _init_embedder(self):
        """Initialize PubMedBERT embedder."""
        if self.embedder is None:
            logger.info("Loading PubMedBERT embedder...")
            from sentence_transformers import SentenceTransformer
            
            model_name = self.config.embedding.model_name
            self.embedder = SentenceTransformer(model_name)
            logger.info(f"Embedder loaded: {model_name}")
    
    def _init_qdrant(self):
        """Initialize Qdrant client."""
        if self.qdrant_client is None:
            logger.info("Connecting to Qdrant...")
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
            import httpx
            
            # Create client
            if self.qdrant_api_key:
                # Cloud connection with API key
                self.qdrant_client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    timeout=120
                )
            elif self.qdrant_username and self.qdrant_password:
                # Self-hosted with Basic Auth
                self.qdrant_client = QdrantClient(
                    url=self.qdrant_url,
                    port=443,
                    timeout=120,
                    prefer_grpc=False,
                    https=True,
                    check_compatibility=False,
                )
                
                # Patch with Basic Auth
                auth = httpx.BasicAuth(self.qdrant_username, self.qdrant_password)
                custom_http = httpx.Client(auth=auth, timeout=120.0)
                
                http_apis = self.qdrant_client._client.http
                for api_name in ['collections_api', 'points_api', 'service_api', 
                                'snapshots_api', 'indexes_api', 'search_api', 
                                'aliases_api', 'distributed_api', 'beta_api']:
                    api = getattr(http_apis, api_name, None)
                    if api and hasattr(api, 'api_client'):
                        api.api_client._client = custom_http
            else:
                # Local connection
                self.qdrant_client = QdrantClient(
                    url=self.qdrant_url or "http://localhost:6333",
                    timeout=120
                )
            
            # Ensure collection exists
            self._ensure_collection()
            
            logger.info(f"Connected to Qdrant, collection: {self.collection_name}")
    
    def _ensure_collection(self):
        """Ensure the ChEMBL collection exists in Qdrant."""
        from qdrant_client.http.models import Distance, VectorParams
        
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.config.qdrant.vector_size,
                    distance=Distance.COSINE
                )
            )
            
            # Create payload indexes for efficient filtering
            self._create_payload_indexes()
            
            logger.info(f"Collection {self.collection_name} created")
        else:
            logger.info(f"Collection {self.collection_name} already exists")
    
    def _create_payload_indexes(self):
        """Create payload indexes for filtering."""
        from qdrant_client.http.models import PayloadSchemaType
        
        indexes = [
            ("target_gene_symbols", PayloadSchemaType.KEYWORD),
            ("chembl_id", PayloadSchemaType.KEYWORD),
            ("drug_name", PayloadSchemaType.TEXT),
            ("indication_names", PayloadSchemaType.TEXT),
            ("approval_status", PayloadSchemaType.KEYWORD),
            ("max_phase", PayloadSchemaType.INTEGER),
            ("action_types", PayloadSchemaType.KEYWORD),
        ]
        
        for field_name, field_type in indexes:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.debug(f"Created index: {field_name}")
            except Exception as e:
                logger.warning(f"Could not create index {field_name}: {e}")
    
    # ==================== Fetch Phase ====================
    
    def fetch_data(
        self,
        limit: Optional[int] = None,
        use_cache: bool = True
    ) -> Tuple[List[Dict], Dict, Dict]:
        """
        Fetch all required data from ChEMBL.
        
        Args:
            limit: Maximum molecules to fetch (None = all approved)
            use_cache: Use cached data if available
        
        Returns:
            Tuple of (molecules, mechanisms_by_molecule, indications_by_molecule)
        """
        self._init_components()
        
        logger.info("=" * 50)
        logger.info("PHASE 1: FETCHING DATA FROM CHEMBL")
        logger.info("=" * 50)
        
        # Fetch molecules
        logger.info("Fetching approved molecules...")
        mol_result = self.molecule_fetcher.fetch_approved_drugs(
            limit=limit,
            use_cache=use_cache
        )
        
        molecules = mol_result.molecules
        chembl_ids = list(mol_result.chembl_ids)
        
        self.stats.total_molecules_fetched = len(molecules)
        logger.info(f"Fetched {len(molecules)} molecules")
        
        # Fetch mechanisms
        logger.info("Fetching mechanisms of action...")
        mech_result = self.mechanism_fetcher.fetch_mechanisms_for_molecules(
            chembl_ids=chembl_ids,
            enrich_with_targets=True,
            use_cache=use_cache
        )
        
        mechanisms_by_molecule = dict(mech_result.mechanisms_by_molecule)
        self.stats.total_mechanisms_fetched = mech_result.total_fetched
        logger.info(f"Fetched {mech_result.total_fetched} mechanisms")
        logger.info(f"Found {len(mech_result.unique_gene_symbols)} unique gene symbols")
        
        # Fetch indications
        logger.info("Fetching drug indications...")
        ind_result = self.indication_fetcher.fetch_indications_for_molecules(
            chembl_ids=chembl_ids,
            use_cache=use_cache
        )
        
        indications_by_molecule = dict(ind_result.indications_by_molecule)
        self.stats.total_indications_fetched = ind_result.total_fetched
        logger.info(f"Fetched {ind_result.total_fetched} indications")
        
        return molecules, mechanisms_by_molecule, indications_by_molecule
    
    # ==================== Normalize Phase ====================
    
    def normalize_data(
        self,
        molecules: List[Dict],
        mechanisms_by_molecule: Dict[str, List[Dict]],
        indications_by_molecule: Dict[str, List[Dict]]
    ) -> List[ChEMBLDrugDocument]:
        """
        Normalize fetched data into documents.
        
        Args:
            molecules: Raw molecule dictionaries
            mechanisms_by_molecule: Mechanisms indexed by ChEMBL ID
            indications_by_molecule: Indications indexed by ChEMBL ID
        
        Returns:
            List of ChEMBLDrugDocument objects
        """
        logger.info("=" * 50)
        logger.info("PHASE 2: NORMALIZING DATA")
        logger.info("=" * 50)
        
        documents, errors = self.normalizer.normalize_batch(
            molecules=molecules,
            mechanisms_by_molecule=mechanisms_by_molecule,
            indications_by_molecule=indications_by_molecule,
            skip_on_error=self.config.ingestion.skip_on_error
        )
        
        self.stats.documents_created = len(documents)
        self.stats.errors = len(errors)
        
        # Calculate additional stats
        gene_symbols = set()
        diseases = set()
        
        for doc in documents:
            if doc.mechanisms:
                self.stats.documents_with_mechanisms += 1
            if doc.indications:
                self.stats.documents_with_indications += 1
            if doc.all_gene_symbols:
                self.stats.documents_with_gene_targets += 1
                gene_symbols.update(doc.all_gene_symbols)
            diseases.update(doc.all_disease_names)
        
        self.stats.unique_gene_symbols = len(gene_symbols)
        self.stats.unique_diseases = len(diseases)
        
        logger.info(f"Created {len(documents)} documents")
        logger.info(f"Documents with mechanisms: {self.stats.documents_with_mechanisms}")
        logger.info(f"Documents with gene targets: {self.stats.documents_with_gene_targets}")
        logger.info(f"Unique gene symbols: {self.stats.unique_gene_symbols}")
        
        return documents
    
    # ==================== Embed & Store Phase ====================
    
    def embed_and_store(
        self,
        documents: List[ChEMBLDrugDocument],
        batch_size: int = 100
    ):
        """
        Generate embeddings and store documents in Qdrant.
        
        Args:
            documents: List of normalized documents
            batch_size: Number of documents to process per batch
        """
        logger.info("=" * 50)
        logger.info("PHASE 3: EMBEDDING AND STORING")
        logger.info("=" * 50)
        
        self._init_embedder()
        self._init_qdrant()
        
        from qdrant_client.http.models import PointStruct
        
        total = len(documents)
        ingested = 0
        
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            
            # Extract text content for embedding
            texts = [doc.text_content for doc in batch]
            
            # Generate embeddings
            logger.info(f"Embedding batch {i//batch_size + 1} ({len(batch)} documents)...")
            embeddings = self.embedder.encode(
                texts,
                show_progress_bar=False,
                batch_size=self.config.embedding.batch_size
            )
            
            # Create points for Qdrant
            points = []
            for doc, embedding in zip(batch, embeddings):
                point = PointStruct(
                    id=doc.doc_id,
                    vector=embedding.tolist(),
                    payload=doc.to_qdrant_payload()
                )
                points.append(point)
            
            # Upsert to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            ingested += len(batch)
            logger.info(f"Ingested {ingested}/{total} documents")
        
        self.stats.documents_ingested = ingested
        logger.info(f"Total documents ingested: {ingested}")
    
    # ==================== Main Pipeline ====================
    
    def run_full_ingestion(
        self,
        limit: Optional[int] = None,
        use_cache: bool = True
    ) -> ChEMBLIngestionStats:
        """
        Run the complete ingestion pipeline.
        
        Args:
            limit: Maximum molecules to process (None = all approved)
            use_cache: Use cached API responses
        
        Returns:
            ChEMBLIngestionStats with ingestion results
        """
        self.stats = ChEMBLIngestionStats()
        self.stats.start_time = datetime.now()
        
        logger.info("=" * 60)
        logger.info("CHEMBL INGESTION PIPELINE STARTED")
        logger.info("=" * 60)
        
        try:
            # Phase 1: Fetch
            molecules, mechanisms, indications = self.fetch_data(
                limit=limit,
                use_cache=use_cache
            )
            
            # Phase 2: Normalize
            documents = self.normalize_data(
                molecules=molecules,
                mechanisms_by_molecule=mechanisms,
                indications_by_molecule=indications
            )
            
            # Phase 3: Embed & Store
            self.embed_and_store(documents)
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            self.stats.errors += 1
            raise
        
        finally:
            self.stats.end_time = datetime.now()
        
        # Log final stats
        logger.info("=" * 60)
        logger.info("CHEMBL INGESTION PIPELINE COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Duration: {self.stats.duration_seconds:.1f} seconds")
        logger.info(f"Molecules fetched: {self.stats.total_molecules_fetched}")
        logger.info(f"Documents created: {self.stats.documents_created}")
        logger.info(f"Documents ingested: {self.stats.documents_ingested}")
        logger.info(f"Unique gene symbols: {self.stats.unique_gene_symbols}")
        logger.info(f"Unique diseases: {self.stats.unique_diseases}")
        logger.info(f"Errors: {self.stats.errors}")
        
        return self.stats
    
    def run_sample_ingestion(self, limit: int = 50) -> ChEMBLIngestionStats:
        """
        Run a sample ingestion for testing.
        
        Args:
            limit: Number of molecules to process
        
        Returns:
            ChEMBLIngestionStats
        """
        logger.info(f"Running sample ingestion with limit={limit}")
        return self.run_full_ingestion(limit=limit, use_cache=True)
    
    # ==================== Utility Methods ====================
    
    def test_connection(self) -> bool:
        """Test connections to ChEMBL API and Qdrant."""
        self._init_components()
        
        # Test ChEMBL
        logger.info("Testing ChEMBL API connection...")
        chembl_ok = self.api_client.test_connection()
        
        # Test Qdrant
        logger.info("Testing Qdrant connection...")
        try:
            self._init_qdrant()
            collections = self.qdrant_client.get_collections()
            qdrant_ok = True
            logger.info(f"Qdrant OK - {len(collections.collections)} collections")
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            qdrant_ok = False
        
        return chembl_ok and qdrant_ok
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for the ChEMBL collection."""
        self._init_qdrant()
        
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def delete_collection(self, confirm: bool = False) -> bool:
        """
        Delete the ChEMBL collection.
        
        Args:
            confirm: Must be True to actually delete
        
        Returns:
            True if deleted
        """
        if not confirm:
            logger.warning("Delete not confirmed. Set confirm=True to delete.")
            return False
        
        self._init_qdrant()
        
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def save_stats(self, filepath: Optional[str] = None):
        """Save ingestion stats to file."""
        if filepath is None:
            filepath = self.cache_dir / "ingestion_stats.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Stats saved to {filepath}")
