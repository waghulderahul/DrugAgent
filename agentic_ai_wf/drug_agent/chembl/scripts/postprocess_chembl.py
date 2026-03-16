#!/usr/bin/env python3
"""
ChEMBL Post-Processing Script
=============================

Processes raw ChEMBL molecule data for vector DB ingestion:
1. SMILES validation - filter out entries without canonical_smiles
2. FDA tagging - identify US FDA approved drugs via cross_references
3. Deduplication - compare against existing Qdrant collection
4. Output new drugs ready for embedding

Usage:
    python -m chembl.scripts.postprocess_chembl
    python -m chembl.scripts.postprocess_chembl --dry-run
    python -m chembl.scripts.postprocess_chembl --output new_drugs.json
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FDA indicator sources in cross_references
FDA_INDICATORS = frozenset({
    'dailymed', 'orange book', 'fda', 'nda', 'anda', 'bla',
    'drugs@fda', 'fda approval', 'fdaapproval'
})


@dataclass
class ProcessingStats:
    """Statistics from post-processing."""
    total_input: int = 0
    valid_smiles: int = 0
    invalid_smiles: int = 0
    fda_approved: int = 0
    non_fda_approved: int = 0
    already_exists: int = 0
    new_to_ingest: int = 0
    
    def summary(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════╗
║              ChEMBL Post-Processing Summary                  ║
╠══════════════════════════════════════════════════════════════╣
║  Total Input Molecules:          {self.total_input:>6}                      ║
║  ─────────────────────────────────────────────────────────── ║
║  Valid SMILES:                   {self.valid_smiles:>6}                      ║
║  Invalid/Missing SMILES:         {self.invalid_smiles:>6}  (filtered out)    ║
║  ─────────────────────────────────────────────────────────── ║
║  FDA Approved (US):              {self.fda_approved:>6}                      ║
║  Non-FDA (Global only):          {self.non_fda_approved:>6}                      ║
║  ─────────────────────────────────────────────────────────── ║
║  Already in DB:                  {self.already_exists:>6}  (skipped)         ║
║  NEW Drugs to Ingest:            {self.new_to_ingest:>6}  ✓                  ║
╚══════════════════════════════════════════════════════════════╝
"""


def get_canonical_smiles(molecule: Dict[str, Any]) -> Optional[str]:
    """Extract canonical SMILES from molecule structure."""
    structures = molecule.get('molecule_structures') or {}
    return structures.get('canonical_smiles')


def is_fda_approved(molecule: Dict[str, Any]) -> bool:
    """Check if molecule has FDA approval indicators in cross_references."""
    cross_refs = molecule.get('cross_references') or []
    
    for ref in cross_refs:
        src = (ref.get('xref_src') or '').lower()
        xref_id = (ref.get('xref_id') or '').lower()
        
        # Check source name
        if any(indicator in src for indicator in FDA_INDICATORS):
            return True
        
        # Check xref_id for NDA/ANDA/BLA patterns
        if any(pattern in xref_id for pattern in ('nda', 'anda', 'bla')):
            return True
    
    return False


def load_existing_from_qdrant(
    collection_name: str,
    qdrant_url: Optional[str] = None,
    qdrant_username: Optional[str] = None,
    qdrant_password: Optional[str] = None
) -> Tuple[Set[str], Set[str]]:
    """
    Load existing chembl_ids and SMILES from Qdrant collection.
    
    Returns:
        Tuple of (set of chembl_ids, set of canonical_smiles)
    """
    existing_ids = set()
    existing_smiles = set()
    
    try:
        from qdrant_client import QdrantClient
        import httpx
        
        url = qdrant_url or os.getenv('QDRANT_URL')
        username = qdrant_username or os.getenv('QDRANT_USERNAME')
        password = qdrant_password or os.getenv('QDRANT_PASSWORD')
        
        if not url:
            logger.warning("QDRANT_URL not set, skipping deduplication")
            return existing_ids, existing_smiles
        
        # Connect to Qdrant
        client = QdrantClient(url=url, port=443, https=True, prefer_grpc=False, timeout=120)
        
        # Apply Basic Auth if credentials provided
        if username and password:
            auth = httpx.BasicAuth(username, password)
            custom_http = httpx.Client(auth=auth, timeout=120.0)
            
            http_apis = client._client.http
            for api_name in ['collections_api', 'points_api', 'service_api']:
                api = getattr(http_apis, api_name, None)
                if api and hasattr(api, 'api_client'):
                    api.api_client._client = custom_http
        
        # Check if collection exists
        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            logger.info(f"Collection '{collection_name}' not found, treating as empty")
            return existing_ids, existing_smiles
        
        # Scroll through all points to get existing IDs and SMILES
        logger.info(f"Loading existing drugs from '{collection_name}'...")
        
        offset = None
        batch_size = 100
        
        while True:
            results = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, next_offset = results
            
            for point in points:
                payload = point.payload or {}
                
                # Get chembl_id
                chembl_id = payload.get('chembl_id')
                if chembl_id:
                    existing_ids.add(chembl_id)
                
                # Get SMILES
                smiles = payload.get('canonical_smiles')
                if smiles:
                    existing_smiles.add(smiles)
            
            if next_offset is None:
                break
            offset = next_offset
        
        logger.info(f"Found {len(existing_ids)} existing chembl_ids, {len(existing_smiles)} existing SMILES")
        
    except ImportError:
        logger.warning("qdrant-client not installed, skipping deduplication")
    except Exception as e:
        logger.warning(f"Could not connect to Qdrant: {e}")
    
    return existing_ids, existing_smiles


def process_molecules(
    molecules: List[Dict[str, Any]],
    existing_ids: Set[str],
    existing_smiles: Set[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], ProcessingStats]:
    """
    Process molecules through validation, tagging, and deduplication.
    
    Returns:
        Tuple of (new_drugs, existing_drugs, stats)
    """
    stats = ProcessingStats(total_input=len(molecules))
    new_drugs = []
    existing_drugs = []
    
    for mol in molecules:
        chembl_id = mol.get('molecule_chembl_id', '')
        smiles = get_canonical_smiles(mol)
        
        # Step 1: SMILES validation
        if not smiles:
            stats.invalid_smiles += 1
            continue
        
        stats.valid_smiles += 1
        
        # Step 2: FDA tagging
        fda_flag = is_fda_approved(mol)
        mol['is_fda_approved'] = fda_flag
        
        if fda_flag:
            stats.fda_approved += 1
        else:
            stats.non_fda_approved += 1
        
        # Step 3: Deduplication (by both chembl_id AND smiles)
        is_duplicate = chembl_id in existing_ids or smiles in existing_smiles
        
        if is_duplicate:
            stats.already_exists += 1
            existing_drugs.append(mol)
        else:
            stats.new_to_ingest += 1
            new_drugs.append(mol)
    
    return new_drugs, existing_drugs, stats


def save_results(
    new_drugs: List[Dict[str, Any]],
    existing_drugs: List[Dict[str, Any]],
    stats: ProcessingStats,
    output_dir: Path,
    output_filename: Optional[str] = None
):
    """Save processing results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save new drugs (ready for ingestion)
    new_drugs_file = output_dir / (output_filename or f'new_drugs_to_ingest_{timestamp}.json')
    with open(new_drugs_file, 'w', encoding='utf-8') as f:
        json.dump(new_drugs, f, indent=2, default=str)
    logger.info(f"Saved {len(new_drugs)} new drugs to: {new_drugs_file}")
    
    # Save existing drugs (for reference)
    if existing_drugs:
        existing_file = output_dir / f'existing_drugs_{timestamp}.json'
        with open(existing_file, 'w', encoding='utf-8') as f:
            json.dump(existing_drugs, f, indent=2, default=str)
        logger.info(f"Saved {len(existing_drugs)} existing drugs to: {existing_file}")
    
    # Save stats
    stats_file = output_dir / f'processing_stats_{timestamp}.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(asdict(stats), f, indent=2)
    
    return new_drugs_file


def main():
    parser = argparse.ArgumentParser(description='Post-process ChEMBL molecules for vector DB ingestion')
    parser.add_argument('--input', type=str, help='Input molecules JSON file (default: cache/molecules_phase4.json)')
    parser.add_argument('--output', type=str, help='Output filename for new drugs')
    parser.add_argument('--output-dir', type=str, help='Output directory (default: cache/processed)')
    parser.add_argument('--collection', type=str, default='ChEMBL_drugs', help='Qdrant collection for deduplication')
    parser.add_argument('--dry-run', action='store_true', help='Show stats without saving files')
    parser.add_argument('--skip-dedup', action='store_true', help='Skip deduplication check')
    
    args = parser.parse_args()
    
    # Load environment
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    
    # Determine paths
    cache_dir = Path(__file__).parent.parent / 'cache'
    input_file = Path(args.input) if args.input else cache_dir / 'molecules_phase4.json'
    output_dir = Path(args.output_dir) if args.output_dir else cache_dir / 'processed'
    
    logger.info("=" * 60)
    logger.info("ChEMBL Post-Processing Pipeline")
    logger.info("=" * 60)
    
    # Load input molecules
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Run the ingestion first: python -m chembl.scripts.run_chembl_ingest")
        sys.exit(1)
    
    logger.info(f"Loading molecules from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        molecules = json.load(f)
    
    logger.info(f"Loaded {len(molecules)} molecules")
    
    # Load existing drugs for deduplication
    existing_ids, existing_smiles = set(), set()
    if not args.skip_dedup:
        existing_ids, existing_smiles = load_existing_from_qdrant(args.collection)
    
    # Process molecules
    logger.info("Processing molecules...")
    new_drugs, existing_drugs, stats = process_molecules(molecules, existing_ids, existing_smiles)
    
    # Print summary
    print(stats.summary())
    
    # Save results
    if not args.dry_run:
        output_file = save_results(new_drugs, existing_drugs, stats, output_dir, args.output)
        logger.info(f"\n✓ Ready for ingestion: {output_file}")
    else:
        logger.info("\n[DRY RUN] No files saved")
    
    return new_drugs, stats


if __name__ == '__main__':
    main()
