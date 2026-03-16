"""
ChEMBL Parser Module
====================

Data normalization and document generation for ChEMBL data.

Components:
-----------
- chembl_normalizer: Normalize raw ChEMBL API data
"""

from .chembl_normalizer import ChEMBLNormalizer

__all__ = [
    "ChEMBLNormalizer",
]
