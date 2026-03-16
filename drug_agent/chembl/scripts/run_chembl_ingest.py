#!/usr/bin/env python3
"""
ChEMBL Ingestion Script
=======================

Full pipeline to fetch, normalize, embed, and store ChEMBL drug data in Qdrant.

Usage:
    python run_chembl_ingest.py [--limit N] [--no-cache] [--sample]

Examples:
    # Full ingestion of all approved drugs
    python run_chembl_ingest.py
    
    # Sample ingestion (50 drugs for testing)
    python run_chembl_ingest.py --sample
    
    # Ingest first 500 drugs
    python run_chembl_ingest.py --limit 500
    
    # Test connection only
    python run_chembl_ingest.py --test
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
        description="Ingest ChEMBL drug data into Qdrant"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of drugs to ingest (default: all)"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run sample ingestion (50 drugs)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached data"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test connections only"
    )
    parser.add_argument(
        "--delete-collection",
        action="store_true",
        help="Delete existing collection before ingesting"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show collection statistics"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Import after args parsed
    from chembl.ingestion import ChEMBLIngestion
    
    logger.info("=" * 60)
    logger.info("ChEMBL Ingestion Pipeline")
    logger.info("=" * 60)
    
    # Initialize ingestion pipeline
    ingestion = ChEMBLIngestion(config_path=args.config)
    
    # Stats only mode
    if args.stats_only:
        stats = ingestion.get_collection_stats()
        logger.info("Collection Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        sys.exit(0)
    
    # Test mode
    if args.test:
        logger.info("Testing connections...")
        if ingestion.test_connection():
            logger.info("All connections successful!")
            sys.exit(0)
        else:
            logger.error("Connection test failed!")
            sys.exit(1)
    
    # Delete collection if requested
    if args.delete_collection:
        logger.warning("Deleting existing collection...")
        response = input("Are you sure you want to delete the collection? (yes/no): ")
        if response.lower() == "yes":
            ingestion.delete_collection(confirm=True)
        else:
            logger.info("Delete cancelled")
            sys.exit(0)
    
    # Determine limit
    limit = args.limit
    if args.sample:
        limit = 50
        logger.info("Running sample ingestion (50 drugs)")
    elif limit:
        logger.info(f"Ingesting up to {limit} drugs")
    else:
        logger.info("Ingesting ALL approved drugs (this may take a while)")
    
    # Run ingestion
    try:
        stats = ingestion.run_full_ingestion(
            limit=limit,
            use_cache=not args.no_cache
        )
        
        # Save stats
        ingestion.save_stats()
        
        # Final summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Duration: {stats.duration_seconds:.1f} seconds")
        logger.info(f"Documents ingested: {stats.documents_ingested}")
        logger.info(f"Documents with gene targets: {stats.documents_with_gene_targets}")
        logger.info(f"Unique gene symbols: {stats.unique_gene_symbols}")
        logger.info(f"Unique diseases: {stats.unique_diseases}")
        
        if stats.errors > 0:
            logger.warning(f"Errors encountered: {stats.errors}")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
