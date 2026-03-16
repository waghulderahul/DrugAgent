#!/usr/bin/env python3
"""
Knowledge Graph CSV to Qdrant Ingestion (GPU Optimized)
=======================================================

GPU-accelerated async pipeline with checkpointing for 8M+ rows.

Usage:
    python ingest_kg_csv.py                          # Auto-resume from checkpoint
    python ingest_kg_csv.py --fresh                  # Start fresh, ignore checkpoint
    python ingest_kg_csv.py --validate-only          # Validate CSV schema only
    python ingest_kg_csv.py --device cuda --batch-size 512
"""

import os
import json
import logging
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    input_file: Path
    collection_name: str
    batch_size: int
    chunk_size: int
    embedding_model: str
    vector_size: int
    start_offset: int
    dry_run: bool
    device: str
    num_workers: int
    checkpoint_file: Path
    checkpoint_interval: int
    validate_only: bool


@dataclass
class Checkpoint:
    rows_processed: int
    collection_name: str
    input_file: str
    timestamp: str
    
    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), indent=2))
    
    @classmethod
    def load(cls, path: Path) -> Optional['Checkpoint']:
        if not path.exists():
            return None
        try:
            return cls(**json.loads(path.read_text()))
        except:
            return None


def detect_device(requested: str) -> str:
    """Auto-detect best available device."""
    if requested != 'auto':
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


def generate_point_id(relation: str, x_id: str, y_id: str) -> str:
    return hashlib.md5(f"{relation}_{x_id}_{y_id}".encode()).hexdigest()


def build_text_content(row: Dict[str, Any]) -> str:
    return f"{row.get('x_name', '')} ({row.get('x_type', '')}) {row.get('display_relation', row.get('relation', ''))} {row.get('y_name', '')} ({row.get('y_type', '')})"


def build_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'relation': row.get('relation', ''),
        'display_relation': row.get('display_relation', ''),
        'x_index': row.get('x_index'),
        'x_id': str(row.get('x_id', '')),
        'x_type': row.get('x_type', ''),
        'x_name': row.get('x_name', ''),
        'x_source': row.get('x_source', ''),
        'y_index': row.get('y_index'),
        'y_id': str(row.get('y_id', '')),
        'y_type': row.get('y_type', ''),
        'y_name': row.get('y_name', ''),
        'y_source': row.get('y_source', ''),
        'text_content': build_text_content(row),
        'doc_type': 'kg_edge',
        'data_source': 'KnowledgeGraph',
        'created_at': datetime.now().isoformat()
    }


def get_qdrant_client(pooled: bool = True):
    """Initialize Qdrant client with connection pooling."""
    from qdrant_client import QdrantClient
    import httpx
    
    url = os.getenv('QDRANT_URL')
    if not url:
        raise ValueError("QDRANT_URL not set")
    
    client = QdrantClient(url=url, port=443, https=True, prefer_grpc=False, timeout=300)
    
    username, password = os.getenv('QDRANT_USERNAME'), os.getenv('QDRANT_PASSWORD')
    if username and password:
        limits = httpx.Limits(max_connections=20, max_keepalive_connections=10) if pooled else None
        custom_http = httpx.Client(auth=httpx.BasicAuth(username, password), timeout=300.0, limits=limits)
        for api_name in ['collections_api', 'points_api', 'service_api']:
            api = getattr(client._client.http, api_name, None)
            if api and hasattr(api, 'api_client'):
                api.api_client._client = custom_http
    
    return client


def ensure_collection(client, collection_name: str, vector_size: int):
    from qdrant_client.models import VectorParams, Distance
    
    if collection_name not in [c.name for c in client.get_collections().collections]:
        logger.info(f"Creating collection: {collection_name}")
        client.create_collection(collection_name=collection_name,
                                  vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))
    else:
        logger.info(f"Collection exists: {collection_name}")


def get_embedding_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer
    actual_device = detect_device(device)
    logger.info(f"Loading {model_name} on {actual_device.upper()}")
    return SentenceTransformer(model_name, device=actual_device), actual_device


def count_csv_rows(file_path: Path) -> int:
    return sum(len(chunk) for chunk in pd.read_csv(file_path, chunksize=100000, usecols=[0]))


def validate_csv(file_path: Path) -> Dict[str, Any]:
    required_cols = {'relation', 'x_id', 'x_name', 'x_type', 'y_id', 'y_name', 'y_type'}
    logger.info(f"Validating: {file_path}")
    
    sample = pd.read_csv(file_path, nrows=5, dtype=str)
    missing = required_cols - set(sample.columns)
    
    if missing:
        return {'valid': False, 'missing_columns': list(missing)}
    
    logger.info("Counting rows...")
    total_rows = count_csv_rows(file_path)
    relations = pd.read_csv(file_path, nrows=100000, dtype=str)['relation'].value_counts().head(10).to_dict()
    
    return {'valid': True, 'total_rows': total_rows, 'columns': list(sample.columns), 'sample_relations': relations}


def upsert_worker(client, collection_name: str, points: List, dry_run: bool, max_retries: int = 3) -> int:
    if dry_run or not points:
        return len(points)
    
    import time
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=collection_name, points=points)
            return len(points)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                logger.warning(f"Upsert failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Upsert failed after {max_retries} attempts: {e}")
                raise
    return 0


def run_ingestion(config: IngestionConfig):
    from qdrant_client.models import PointStruct
    
    logger.info("=" * 60)
    logger.info(f"KG Ingestion | {config.input_file.name} → {config.collection_name}")
    logger.info(f"Batch: {config.batch_size} | Workers: {config.num_workers} | Device: {config.device}")
    logger.info("=" * 60)
    
    if not config.input_file.exists():
        logger.error(f"Input file not found: {config.input_file}")
        return
    
    if config.validate_only:
        print(json.dumps(validate_csv(config.input_file), indent=2))
        return
    
    # Auto-resume from checkpoint
    checkpoint = Checkpoint.load(config.checkpoint_file)
    start_offset = config.start_offset
    if checkpoint and checkpoint.input_file == str(config.input_file):
        start_offset = max(start_offset, checkpoint.rows_processed)
        logger.info(f"Resuming from checkpoint: row {start_offset:,}")
    
    # Initialize
    client = get_qdrant_client(pooled=True)
    ensure_collection(client, config.collection_name, config.vector_size)
    model, actual_device = get_embedding_model(config.embedding_model, config.device)
    
    logger.info("Counting total rows...")
    total_rows = count_csv_rows(config.input_file)
    rows_to_process = total_rows - start_offset
    logger.info(f"Total: {total_rows:,} | To process: {rows_to_process:,}")
    
    total_ingested, rows_read = 0, 0
    start_time = datetime.now()
    executor = ThreadPoolExecutor(max_workers=config.num_workers)
    pending_futures = []
    pbar = tqdm(total=rows_to_process, desc="Ingesting", unit="edges", dynamic_ncols=True, smoothing=0.1)
    
    try:
        batch = []
        for chunk in pd.read_csv(config.input_file, chunksize=config.chunk_size, dtype=str):
            for _, row in chunk.iterrows():
                rows_read += 1
                if rows_read <= start_offset:
                    continue
                
                batch.append(row.to_dict())
                
                if len(batch) >= config.batch_size:
                    # GPU-accelerated embedding
                    payloads = [build_payload(r) for r in batch]
                    embeddings = model.encode([p['text_content'] for p in payloads], 
                                              show_progress_bar=False, convert_to_numpy=True)
                    
                    points = [
                        PointStruct(id=generate_point_id(r.get('relation', ''), str(r.get('x_id', '')), str(r.get('y_id', ''))),
                                    vector=emb.tolist(), payload=pay)
                        for r, pay, emb in zip(batch, payloads, embeddings)
                    ]
                    
                    # Async upsert
                    pending_futures.append(executor.submit(upsert_worker, client, config.collection_name, points, config.dry_run))
                    
                    # Collect completed
                    for f in [f for f in pending_futures if f.done()]:
                        total_ingested += f.result()
                        pending_futures.remove(f)
                    
                    pbar.update(len(batch))
                    batch = []
                    
                    # Checkpoint
                    if rows_read % config.checkpoint_interval == 0:
                        Checkpoint(rows_read, config.collection_name, str(config.input_file), 
                                   datetime.utcnow().isoformat()).save(config.checkpoint_file)
        
        # Final batch
        if batch:
            payloads = [build_payload(r) for r in batch]
            embeddings = model.encode([p['text_content'] for p in payloads], show_progress_bar=False, convert_to_numpy=True)
            points = [
                PointStruct(id=generate_point_id(r.get('relation', ''), str(r.get('x_id', '')), str(r.get('y_id', ''))),
                            vector=emb.tolist(), payload=pay)
                for r, pay, emb in zip(batch, payloads, embeddings)
            ]
            pending_futures.append(executor.submit(upsert_worker, client, config.collection_name, points, config.dry_run))
            pbar.update(len(batch))
        
        for f in as_completed(pending_futures):
            total_ingested += f.result()
    finally:
        pbar.close()
        executor.shutdown(wait=True)
    
    # Final checkpoint
    Checkpoint(rows_read, config.collection_name, str(config.input_file), datetime.utcnow().isoformat()).save(config.checkpoint_file)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info(f"DONE | {total_ingested:,} edges | {elapsed:.1f}s | {total_ingested/max(elapsed,1):.1f}/sec")
    if not config.dry_run:
        logger.info(f"Collection points: {client.get_collection(config.collection_name).points_count:,}")


def main():
    parser = argparse.ArgumentParser(description='Ingest KG CSV to Qdrant (GPU Optimized)')
    parser.add_argument('--input', default='kg.csv')
    parser.add_argument('--collection', default='Raw_csv_KG')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--chunk-size', type=int, default=100000)
    parser.add_argument('--start-offset', type=int, default=0)
    parser.add_argument('--embedding-model', default='NeuML/pubmedbert-base-embeddings')
    parser.add_argument('--vector-size', type=int, default=768)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--checkpoint-file', default='.kg_ingest_checkpoint')
    parser.add_argument('--checkpoint-interval', type=int, default=10000)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--fresh', action='store_true')
    
    args = parser.parse_args()
    load_dotenv(Path(__file__).parent.parent / '.env')
    
    input_file = Path(args.input) if Path(args.input).is_absolute() else Path(__file__).parent / args.input
    checkpoint_file = Path(args.checkpoint_file) if Path(args.checkpoint_file).is_absolute() else Path(__file__).parent / args.checkpoint_file
    
    if args.fresh and checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("Removed checkpoint, starting fresh")
    
    run_ingestion(IngestionConfig(
        input_file=input_file, collection_name=args.collection, batch_size=args.batch_size,
        chunk_size=args.chunk_size, embedding_model=args.embedding_model, vector_size=args.vector_size,
        start_offset=args.start_offset, dry_run=args.dry_run, device=args.device,
        num_workers=args.num_workers, checkpoint_file=checkpoint_file,
        checkpoint_interval=args.checkpoint_interval, validate_only=args.validate_only
    ))


if __name__ == '__main__':
    main()
