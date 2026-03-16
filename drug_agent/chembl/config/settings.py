"""
ChEMBL Configuration Settings
=============================

Load and manage configuration for ChEMBL integration.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ChEMBLAPIConfig:
    """ChEMBL API configuration."""
    base_url: str = "https://www.ebi.ac.uk/chembl/api/data"
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay_seconds: int = 2
    batch_size: int = 100
    rate_limit: int = 10


@dataclass
class FetchingConfig:
    """Data fetching configuration."""
    max_phase: int = 4
    fetch_mechanisms: bool = True
    fetch_indications: bool = True
    fetch_targets: bool = True
    fetch_bioactivities: bool = False
    molecule_types: Optional[List[str]] = None
    max_drugs: Optional[int] = None
    require_mechanism: bool = False


@dataclass
class CacheConfig:
    """Caching configuration."""
    enabled: bool = True
    directory: str = "./cache"
    expiry_days: int = 30
    files: Dict[str, str] = field(default_factory=lambda: {
        "molecules": "molecules_cache.json",
        "mechanisms": "mechanisms_cache.json",
        "indications": "indications_cache.json",
        "targets": "targets_cache.json"
    })


@dataclass
class QdrantConfig:
    """Qdrant storage configuration for ChEMBL."""
    collection_name: str = "ChEMBL_drugs"
    vector_size: int = 768
    distance_metric: str = "Cosine"
    use_parent_connection: bool = True
    upsert_batch_size: int = 100
    payload_indexes: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    use_parent_embedder: bool = True
    model_name: str = "NeuML/pubmedbert-base-embeddings"
    batch_size: int = 32
    cache_enabled: bool = True
    cache_file: str = "./cache/chembl_embeddings_cache.json"


@dataclass
class IngestionConfig:
    """Ingestion pipeline configuration."""
    show_progress: bool = True
    checkpoint_frequency: int = 500
    checkpoint_file: str = "./cache/ingestion_checkpoint.json"
    resume_from_checkpoint: bool = True
    max_workers: int = 4
    validate_data: bool = True
    skip_on_error: bool = True


@dataclass
class SearchConfig:
    """Search configuration for unified search."""
    collections: List[str] = field(default_factory=lambda: ["Drug_agent", "ChEMBL_drugs"])
    results_per_collection: int = 30
    final_top_k: int = 20
    min_relevance_score: float = 0.3
    merge_strategy: str = "union"
    dedup_field: str = "drug_name"
    ranking_weights: Dict[str, float] = field(default_factory=lambda: {
        "relevance": 0.30,
        "gene_match": 0.35,
        "mechanism_evidence": 0.20,
        "approval_status": 0.15
    })


@dataclass
class ChEMBLConfig:
    """Main ChEMBL configuration container."""
    chembl_api: ChEMBLAPIConfig = field(default_factory=ChEMBLAPIConfig)
    fetching: FetchingConfig = field(default_factory=FetchingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChEMBLConfig":
        """Create config from dictionary."""
        return cls(
            chembl_api=ChEMBLAPIConfig(**data.get("chembl_api", {})),
            fetching=FetchingConfig(**data.get("fetching", {})),
            cache=CacheConfig(**data.get("cache", {})),
            qdrant=QdrantConfig(**data.get("qdrant", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            ingestion=IngestionConfig(**data.get("ingestion", {})),
            search=SearchConfig(**data.get("search", {}))
        )


def load_chembl_config(config_path: Optional[str] = None) -> ChEMBLConfig:
    """
    Load ChEMBL configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
    
    Returns:
        ChEMBLConfig object
    """
    if config_path is None:
        # Default config path
        config_path = Path(__file__).parent / "chembl_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return ChEMBLConfig()
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if data is None:
        return ChEMBLConfig()
    
    return ChEMBLConfig.from_dict(data)


def get_cache_dir(config: ChEMBLConfig) -> Path:
    """Get absolute path to cache directory."""
    cache_dir = Path(config.cache.directory)
    if not cache_dir.is_absolute():
        # Relative to chembl module
        cache_dir = Path(__file__).parent.parent / cache_dir
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
