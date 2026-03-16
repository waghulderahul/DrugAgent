"""
ChEMBL Ingestion Module
=======================

Orchestration of the ChEMBL data ingestion pipeline.

Components:
-----------
- chembl_ingest: Main ingestion orchestrator
"""

from .chembl_ingest import ChEMBLIngestion

__all__ = [
    "ChEMBLIngestion",
]
