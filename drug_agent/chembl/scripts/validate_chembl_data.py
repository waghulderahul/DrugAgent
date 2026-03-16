#!/usr/bin/env python3
"""
ChEMBL Data Validation Script
=============================

Validate ingested ChEMBL data by running test queries.

Usage:
    python validate_chembl_data.py [--gene GENE] [--drug DRUG] [--disease DISEASE]

Examples:
    # Run all validation tests
    python validate_chembl_data.py
    
    # Search by gene
    python validate_chembl_data.py --gene EGFR
    
    # Search by drug name
    python validate_chembl_data.py --drug Imatinib
    
    # Search by disease
    python validate_chembl_data.py --disease "lung cancer"
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_qdrant_client():
    """Initialize Qdrant client with auth."""
    from qdrant_client import QdrantClient
    import httpx
    
    url = os.getenv("QDRANT_URL")
    username = os.getenv("QDRANT_USERNAME")
    password = os.getenv("QDRANT_PASSWORD")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if api_key:
        return QdrantClient(url=url, api_key=api_key, timeout=60)
    elif username and password:
        client = QdrantClient(
            url=url,
            port=443,
            timeout=60,
            prefer_grpc=False,
            https=True,
            check_compatibility=False,
        )
        
        auth = httpx.BasicAuth(username, password)
        custom_http = httpx.Client(auth=auth, timeout=60.0)
        
        http_apis = client._client.http
        for api_name in ['collections_api', 'points_api', 'search_api']:
            api = getattr(http_apis, api_name, None)
            if api and hasattr(api, 'api_client'):
                api.api_client._client = custom_http
        
        return client
    else:
        return QdrantClient(url=url or "http://localhost:6333", timeout=60)


def get_embedder():
    """Load PubMedBERT embedder."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("NeuML/pubmedbert-base-embeddings")


def search_by_vector(
    client,
    embedder,
    query: str,
    collection: str = "ChEMBL_drugs",
    limit: int = 10,
    filter_dict: dict = None
):
    """Search collection by vector similarity."""
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    
    # Generate query embedding
    query_vector = embedder.encode(query).tolist()
    
    # Build filter if provided
    qdrant_filter = None
    if filter_dict:
        conditions = []
        for key, value in filter_dict.items():
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )
        qdrant_filter = Filter(must=conditions)
    
    # Search using query_points for newer qdrant-client
    from qdrant_client.models import QueryRequest
    try:
        results = client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=limit
        ).points
    except Exception:
        # Fallback for older API
        results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=limit
        )
    
    return results


def print_results(results, title: str):
    """Pretty print search results."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)
    
    if not results:
        logger.info("No results found")
        return
    
    for i, result in enumerate(results, 1):
        payload = result.payload
        logger.info(f"\n{i}. {payload.get('drug_name', 'Unknown')} ({payload.get('chembl_id', '')})")
        logger.info(f"   Score: {result.score:.4f}")
        logger.info(f"   Status: {payload.get('approval_status', 'Unknown')}")
        
        genes = payload.get('target_gene_symbols', [])
        if genes:
            logger.info(f"   Target Genes: {', '.join(genes[:5])}")
        
        mech = payload.get('mechanism_of_action', '')
        if mech:
            logger.info(f"   Mechanism: {mech[:80]}...")
        
        diseases = payload.get('indication_names', [])
        if diseases:
            logger.info(f"   Indications: {', '.join(diseases[:3])}")


def run_validation_tests(client, embedder, collection: str):
    """Run a set of validation tests."""
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING VALIDATION TESTS")
    logger.info("=" * 60)
    
    tests = [
        {
            "name": "Gene Target: EGFR",
            "query": "drugs targeting EGFR gene for cancer treatment",
            "filter": {"target_gene_symbols": "EGFR"}
        },
        {
            "name": "Gene Target: HRH1 (Histamine receptor)",
            "query": "antihistamine drugs targeting HRH1",
            "filter": {"target_gene_symbols": "HRH1"}
        },
        {
            "name": "Kinase Inhibitors",
            "query": "kinase inhibitor cancer drugs",
            "filter": None
        },
        {
            "name": "Approved Drugs",
            "query": "FDA approved therapeutic drugs",
            "filter": {"approval_status": "FDA Approved"}
        },
    ]
    
    for test in tests:
        try:
            results = search_by_vector(
                client=client,
                embedder=embedder,
                query=test["query"],
                collection=collection,
                limit=5,
                filter_dict=test.get("filter")
            )
            print_results(results, f"Test: {test['name']}")
        except Exception as e:
            logger.error(f"Test '{test['name']}' failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate ChEMBL data in Qdrant"
    )
    parser.add_argument(
        "--gene",
        type=str,
        help="Search for drugs targeting a specific gene"
    )
    parser.add_argument(
        "--drug",
        type=str,
        help="Search for a specific drug by name"
    )
    parser.add_argument(
        "--disease",
        type=str,
        help="Search for drugs for a disease"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="ChEMBL_drugs",
        help="Qdrant collection name"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum results to return"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection statistics only"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("ChEMBL Data Validator")
    logger.info("=" * 60)
    
    # Initialize clients
    logger.info("Connecting to Qdrant...")
    client = get_qdrant_client()
    
    # Show stats
    if args.stats:
        try:
            info = client.get_collection(args.collection)
            logger.info(f"\nCollection: {args.collection}")
            logger.info(f"Points: {info.points_count}")
            logger.info(f"Vectors: {info.vectors_count}")
            logger.info(f"Status: {info.status}")
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
        sys.exit(0)
    
    # Load embedder
    logger.info("Loading embedder...")
    embedder = get_embedder()
    
    # Run specific query or validation tests
    if args.gene:
        query = f"drugs targeting {args.gene} gene"
        results = search_by_vector(
            client, embedder, query, args.collection, args.limit,
            filter_dict={"target_gene_symbols": args.gene.upper()}
        )
        print_results(results, f"Drugs targeting gene: {args.gene}")
    
    elif args.drug:
        query = f"{args.drug} drug mechanism and indications"
        results = search_by_vector(
            client, embedder, query, args.collection, args.limit
        )
        print_results(results, f"Search for drug: {args.drug}")
    
    elif args.disease:
        query = f"approved drugs for {args.disease} treatment"
        results = search_by_vector(
            client, embedder, query, args.collection, args.limit
        )
        print_results(results, f"Drugs for disease: {args.disease}")
    
    else:
        # Run all validation tests
        run_validation_tests(client, embedder, args.collection)
    
    logger.info("\n" + "=" * 60)
    logger.info("Validation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
