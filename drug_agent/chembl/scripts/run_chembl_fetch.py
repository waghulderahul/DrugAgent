#!/usr/bin/env python3
"""
ChEMBL Data Fetch Script
========================

Fetch approved drug data from ChEMBL API and cache locally.

Usage:
    python run_chembl_fetch.py [--limit N] [--no-cache] [--clear-cache]

Examples:
    # Fetch all approved drugs (with caching)
    python run_chembl_fetch.py
    
    # Fetch first 100 drugs
    python run_chembl_fetch.py --limit 100
    
    # Fetch without using cache
    python run_chembl_fetch.py --no-cache
    
    # Clear existing cache before fetching
    python run_chembl_fetch.py --clear-cache
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch ChEMBL approved drug data"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of drugs to fetch (default: all)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached data and fetch fresh"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear existing cache before fetching"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test API connection, don't fetch data"
    )
    
    args = parser.parse_args()
    
    # Import after args parsed (faster startup for --help)
    from chembl.fetcher import ChEMBLAPIClient, MoleculeFetcher, MechanismFetcher, IndicationFetcher
    from chembl.config.settings import load_chembl_config, get_cache_dir
    
    config = load_chembl_config()
    cache_dir = get_cache_dir(config)
    
    logger.info("=" * 60)
    logger.info("ChEMBL Data Fetcher")
    logger.info("=" * 60)
    logger.info(f"Cache directory: {cache_dir}")
    
    # Initialize API client
    api_client = ChEMBLAPIClient(
        cache_dir=str(cache_dir),
        cache_enabled=not args.no_cache
    )
    
    # Clear cache if requested
    if args.clear_cache:
        logger.info("Clearing cache...")
        api_client.clear_cache()
    
    # Test connection
    logger.info("Testing ChEMBL API connection...")
    if not api_client.test_connection():
        logger.error("Failed to connect to ChEMBL API")
        sys.exit(1)
    
    if args.test_only:
        logger.info("Connection test successful!")
        sys.exit(0)
    
    # Fetch molecules
    logger.info("-" * 40)
    logger.info("Fetching approved molecules...")
    mol_fetcher = MoleculeFetcher(api_client)
    mol_result = mol_fetcher.fetch_approved_drugs(
        limit=args.limit,
        use_cache=not args.no_cache
    )
    
    logger.info(f"Fetched {mol_result.total_fetched} molecules")
    if mol_result.errors:
        logger.warning(f"Errors: {mol_result.errors}")
    
    # Show molecule statistics
    stats = mol_fetcher.get_molecule_statistics(mol_result.molecules)
    logger.info(f"Molecule types: {stats['molecule_types']}")
    
    # Fetch mechanisms
    logger.info("-" * 40)
    logger.info("Fetching mechanisms of action...")
    mech_fetcher = MechanismFetcher(api_client)
    mech_result = mech_fetcher.fetch_mechanisms_for_molecules(
        chembl_ids=list(mol_result.chembl_ids),
        enrich_with_targets=True,
        use_cache=not args.no_cache
    )
    
    logger.info(f"Fetched {mech_result.total_fetched} mechanisms")
    logger.info(f"Unique gene symbols: {len(mech_result.unique_gene_symbols)}")
    
    # Fetch indications
    logger.info("-" * 40)
    logger.info("Fetching drug indications...")
    ind_fetcher = IndicationFetcher(api_client)
    ind_result = ind_fetcher.fetch_indications_for_molecules(
        chembl_ids=list(mol_result.chembl_ids),
        use_cache=not args.no_cache
    )
    
    logger.info(f"Fetched {ind_result.total_fetched} indications")
    logger.info(f"Unique diseases: {len(ind_result.unique_disease_names)}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("FETCH SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Molecules:   {mol_result.total_fetched}")
    logger.info(f"Mechanisms:  {mech_result.total_fetched}")
    logger.info(f"Indications: {ind_result.total_fetched}")
    logger.info(f"Gene symbols: {len(mech_result.unique_gene_symbols)}")
    logger.info(f"Diseases:    {len(ind_result.unique_disease_names)}")
    logger.info(f"Cache dir:   {cache_dir}")
    logger.info("=" * 60)
    
    # Show sample gene symbols
    if mech_result.unique_gene_symbols:
        sample_genes = list(mech_result.unique_gene_symbols)[:20]
        logger.info(f"Sample gene symbols: {', '.join(sample_genes)}")


if __name__ == "__main__":
    main()
