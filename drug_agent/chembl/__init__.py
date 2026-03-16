"""
ChEMBL Integration Module
=========================

This module provides functionality for fetching, parsing, and ingesting
drug data from the ChEMBL database into the Drug Discovery Agent.

Submodules:
-----------
- config: Configuration management for ChEMBL integration
- models: Data models for ChEMBL drug documents
- fetcher: API clients for fetching data from ChEMBL
- parser: Data normalization and document generation
- ingestion: Orchestration of the ingestion pipeline

Usage:
------
    from chembl import ChEMBLIngestion
    
    ingestion = ChEMBLIngestion()
    stats = ingestion.run_full_ingestion()
    print(f"Ingested {stats.documents_ingested} drugs")

For more details, see CHEMBL_INTEGRATION_GUIDE.md
"""

__version__ = "1.0.0"
__author__ = "Drug Discovery Agent Team"

# Import main classes for convenience
from .ingestion.chembl_ingest import ChEMBLIngestion
from .models.chembl_models import ChEMBLDrugDocument, ChEMBLIngestionStats
from .fetcher.chembl_api_client import ChEMBLAPIClient
from .parser.chembl_normalizer import ChEMBLNormalizer

__all__ = [
    "__version__",
    "ChEMBLIngestion",
    "ChEMBLDrugDocument",
    "ChEMBLIngestionStats",
    "ChEMBLAPIClient",
    "ChEMBLNormalizer",
]
