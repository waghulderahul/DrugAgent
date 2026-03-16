# ChEMBL Drug Data Integration Guide

## Complete Documentation for Integrating ChEMBL into Drug Discovery Agent

**Version:** 1.0.0  
**Date:** January 19, 2026  
**Author:** Drug Discovery Agent Team

---

## Table of Contents

1. [Overview](#1-overview)
2. [ChEMBL Data Source](#2-chembl-data-source)
3. [Architecture Design](#3-architecture-design)
4. [Data Schema](#4-data-schema)
5. [Implementation Phases](#5-implementation-phases)
6. [Module Specifications](#6-module-specifications)
7. [API Reference](#7-api-reference)
8. [Storage Strategy](#8-storage-strategy)
9. [Search & Retrieval](#9-search--retrieval)
10. [Testing Plan](#10-testing-plan)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Overview

### 1.1 Purpose

This document describes the integration of ChEMBL drug database into the existing Drug Discovery Agent. The goal is to enrich the knowledge base with comprehensive drug information including:

- Approved drug details
- Mechanism of action
- Target gene associations
- Disease indications
- Bioactivity data

### 1.2 Current State

The Drug Discovery Agent currently ingests gene-centric data from JSON files (GeneALaCart) containing:
- Gene-drug associations
- Gene-disease relationships
- Pathway information

**Limitation:** The current data is gene-centric, meaning we find drugs THROUGH genes.

### 1.3 Target State

After ChEMBL integration:
- Drug-centric data available directly
- Explicit mechanism of action information
- Validated target-gene mappings
- Clinical approval status with evidence
- Bidirectional lookup: Gene → Drug AND Drug → Gene

### 1.4 Key Benefits

| Benefit | Description |
|---------|-------------|
| **Richer Drug Data** | ChEMBL has 15,591 drugs with detailed annotations |
| **Mechanism Context** | Know HOW a drug works (inhibitor, antagonist, etc.) |
| **Gene Linking** | Explicit target_gene_symbol for direct matching |
| **Clinical Evidence** | Approval phase and bioactivity measurements |
| **Disease Indications** | Direct drug-disease associations from clinical data |

---

## 2. ChEMBL Data Source

### 2.1 About ChEMBL

ChEMBL is a manually curated database of bioactive molecules with drug-like properties, maintained by the European Bioinformatics Institute (EMBL-EBI).

- **Website:** https://www.ebi.ac.uk/chembl/
- **API Docs:** https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services
- **Python Client:** `chembl_webresource_client`

### 2.2 Data Statistics

| Metric | Count |
|--------|-------|
| Total Compounds | ~2.4 million |
| Total Drugs | 15,591 |
| Approved Drugs (Phase 4) | ~2,795 |
| Drugs with Mechanisms | ~2,500 |
| Drugs with Gene Targets | ~2,200 |
| Drug Indications | ~15,000+ |

### 2.3 Data Access Method

**Chosen Approach:** ChEMBL Web API with Python Client

```bash
pip install chembl_webresource_client
```

**Why API over Bulk Download:**
1. ✅ Official Python client with clean interface
2. ✅ Paginated access (handles large datasets)
3. ✅ Filter support (only fetch approved drugs)
4. ✅ Always up-to-date data
5. ✅ No large file storage needed

### 2.4 Key API Endpoints

| Endpoint | Purpose | Example |
|----------|---------|---------|
| `/molecule` | Drug/compound details | `molecule.filter(max_phase=4)` |
| `/mechanism` | Mechanism of action | `mechanism.filter(molecule_chembl_id=X)` |
| `/drug_indication` | Disease indications | `drug_indication.filter(molecule_chembl_id=X)` |
| `/target` | Target protein details | `target.filter(target_chembl_id=X)` |
| `/activity` | Bioactivity measurements | `activity.filter(molecule_chembl_id=X)` |

### 2.5 Rate Limiting & Best Practices

- ChEMBL API has no strict rate limits but be respectful
- Use batch requests where possible
- Implement exponential backoff for retries
- Cache responses to avoid redundant calls
- Process in batches of 100-500 molecules

---

## 3. Architecture Design

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DRUG DISCOVERY AGENT                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────┐      ┌──────────────────────┐            │
│  │   EXISTING PIPELINE  │      │   NEW CHEMBL PIPELINE │            │
│  │                      │      │                       │            │
│  │  ┌────────────────┐  │      │  ┌─────────────────┐  │            │
│  │  │ Gene JSON      │  │      │  │ ChEMBL API      │  │            │
│  │  │ (GeneALaCart)  │  │      │  │ (Web Service)   │  │            │
│  │  └───────┬────────┘  │      │  └────────┬────────┘  │            │
│  │          │           │      │           │           │            │
│  │  ┌───────▼────────┐  │      │  ┌────────▼────────┐  │            │
│  │  │ json_parser.py │  │      │  │ chembl_fetcher  │  │            │
│  │  └───────┬────────┘  │      │  └────────┬────────┘  │            │
│  │          │           │      │           │           │            │
│  │  ┌───────▼────────┐  │      │  ┌────────▼────────┐  │            │
│  │  │ document_gen   │  │      │  │ chembl_parser   │  │            │
│  │  └───────┬────────┘  │      │  └────────┬────────┘  │            │
│  │          │           │      │           │           │            │
│  └──────────┼───────────┘      └───────────┼──────────┘            │
│             │                              │                        │
│             │    ┌──────────────────┐      │                        │
│             └───►│  PubMedBERT      │◄─────┘                        │
│                  │  Embedder        │                               │
│                  └────────┬─────────┘                               │
│                           │                                         │
│             ┌─────────────┴─────────────┐                          │
│             │                           │                          │
│             ▼                           ▼                          │
│  ┌─────────────────────┐    ┌─────────────────────┐               │
│  │   QDRANT COLLECTION │    │   QDRANT COLLECTION │               │
│  │   "Drug_agent"      │    │   "ChEMBL_drugs"    │               │
│  │   (Gene-centric)    │    │   (Drug-centric)    │               │
│  └──────────┬──────────┘    └──────────┬──────────┘               │
│             │                          │                           │
│             └───────────┬──────────────┘                           │
│                         │                                          │
│                         ▼                                          │
│              ┌─────────────────────┐                               │
│              │   UNIFIED SEARCH    │                               │
│              │   (Merge & Rerank)  │                               │
│              └─────────────────────┘                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Folder Structure

```
drug_discovery_agent/
├── chembl/                          # NEW CHEMBL MODULE
│   ├── __init__.py
│   ├── CHEMBL_INTEGRATION_GUIDE.md  # This document
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── chembl_config.yaml       # ChEMBL-specific configuration
│   │
│   ├── fetcher/
│   │   ├── __init__.py
│   │   ├── chembl_api_client.py     # API wrapper with retry logic
│   │   ├── molecule_fetcher.py      # Fetch approved molecules
│   │   ├── mechanism_fetcher.py     # Fetch mechanisms of action
│   │   └── indication_fetcher.py    # Fetch drug indications
│   │
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── chembl_normalizer.py     # Normalize ChEMBL data
│   │   └── document_generator.py    # Generate embedding documents
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── chembl_models.py         # Pydantic/dataclass models
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chembl_ingest.py         # Main ingestion orchestrator
│   │   └── batch_processor.py       # Batch embedding & upload
│   │
│   ├── cache/
│   │   └── .gitkeep                 # Cache directory for API responses
│   │
│   └── scripts/
│       ├── run_chembl_fetch.py      # CLI: Fetch ChEMBL data
│       ├── run_chembl_ingest.py     # CLI: Ingest to Qdrant
│       └── validate_chembl_data.py  # CLI: Validate ingested data
│
├── retrieval/
│   ├── hybrid_search.py             # MODIFIED: Add multi-collection search
│   └── unified_search.py            # NEW: Merge results from both collections
│
└── ... (existing modules)
```

### 3.3 Data Flow

```
Step 1: FETCH
┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ ChEMBL API  │───►│ chembl_fetcher  │───►│ Raw JSON/Cache   │
└─────────────┘    └─────────────────┘    └──────────────────┘

Step 2: PARSE
┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Raw JSON/Cache   │───►│ chembl_parser   │───►│ Normalized Docs  │
└──────────────────┘    └─────────────────┘    └──────────────────┘

Step 3: EMBED
┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Normalized Docs  │───►│ PubMedBERT      │───►│ Vectors + Meta   │
└──────────────────┘    └─────────────────┘    └──────────────────┘

Step 4: STORE
┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Vectors + Meta   │───►│ Qdrant Client   │───►│ ChEMBL_drugs     │
└──────────────────┘    └─────────────────┘    │ Collection       │
                                               └──────────────────┘

Step 5: QUERY (Runtime)
┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ User Query       │───►│ Unified Search  │───►│ Drug_agent +     │
│ (Gene/Disease)   │    │                 │    │ ChEMBL_drugs     │
└──────────────────┘    └─────────────────┘    └──────────────────┘
```

---

## 4. Data Schema

### 4.1 ChEMBL Raw Data (From API)

**Molecule Response:**
```json
{
    "molecule_chembl_id": "CHEMBL998",
    "pref_name": "LORATADINE",
    "max_phase": 4,
    "molecule_type": "Small molecule",
    "first_approval": 1993,
    "oral": true,
    "parenteral": false,
    "topical": false,
    "molecule_synonyms": [
        {"synonym": "Claritin", "syn_type": "TRADE_NAME"},
        {"synonym": "Claratyne", "syn_type": "TRADE_NAME"}
    ],
    "molecule_properties": {
        "molecular_weight": "382.89",
        "alogp": "4.55",
        "hba": 3,
        "hbd": 0
    }
}
```

**Mechanism Response:**
```json
{
    "molecule_chembl_id": "CHEMBL998",
    "mechanism_of_action": "Histamine H1 receptor antagonist",
    "target_chembl_id": "CHEMBL231",
    "action_type": "ANTAGONIST",
    "direct_interaction": true,
    "disease_efficacy": true,
    "target_name": "Histamine H1 receptor",
    "target_type": "SINGLE PROTEIN",
    "target_organism": "Homo sapiens",
    "target_components": [
        {
            "accession": "P35367",
            "component_type": "PROTEIN",
            "gene_symbol": "HRH1"
        }
    ]
}
```

**Drug Indication Response:**
```json
{
    "molecule_chembl_id": "CHEMBL998",
    "mesh_id": "D006255",
    "mesh_heading": "Rhinitis, Allergic, Seasonal",
    "efo_id": "EFO_0003931",
    "efo_term": "allergic rhinitis",
    "max_phase_for_ind": 4,
    "indication_refs": [
        {"ref_type": "DailyMed", "ref_id": "..."}
    ]
}
```

### 4.2 Normalized ChEMBL Document (For Embedding)

```python
@dataclass
class ChEMBLDrugDocument:
    """Normalized document for Qdrant storage."""
    
    # Identifiers
    doc_id: str                      # UUID for Qdrant
    chembl_id: str                   # CHEMBL998
    drug_name: str                   # LORATADINE
    synonyms: List[str]              # [Claritin, Claratyne]
    
    # Approval Status
    max_phase: int                   # 4 = Approved
    first_approval: Optional[int]   # 1993
    approval_status: str            # "FDA Approved"
    
    # Drug Properties
    molecule_type: str              # "Small molecule"
    oral: bool
    parenteral: bool
    topical: bool
    molecular_weight: Optional[float]
    
    # Mechanisms (CRITICAL for gene linking)
    mechanisms: List[MechanismInfo]
    """
    MechanismInfo:
        - mechanism_of_action: str      # "Histamine H1 receptor antagonist"
        - action_type: str              # "ANTAGONIST"
        - target_chembl_id: str         # "CHEMBL231"
        - target_name: str              # "Histamine H1 receptor"
        - target_gene_symbol: str       # "HRH1" ← KEY FOR GENE LINKING
        - target_uniprot: str           # "P35367"
    """
    
    # Indications (for disease matching)
    indications: List[IndicationInfo]
    """
    IndicationInfo:
        - mesh_id: str                  # "D006255"
        - mesh_heading: str             # "Rhinitis, Allergic"
        - efo_id: str                   # "EFO_0003931"
        - efo_term: str                 # "allergic rhinitis"
        - max_phase_for_ind: int        # 4
    """
    
    # Derived fields for search
    all_gene_symbols: List[str]     # All target genes
    all_disease_names: List[str]    # All indication names
    
    # Text content for embedding
    text_content: str               # Concatenated searchable text
    
    # Metadata
    doc_type: str = "chembl_drug"
    data_source: str = "ChEMBL"
    created_at: datetime
```

### 4.3 Text Content Generation

The `text_content` field is what gets embedded. It should be rich and searchable:

```python
def generate_text_content(drug: ChEMBLDrugDocument) -> str:
    """Generate embedding-optimized text content."""
    
    parts = []
    
    # Drug identification
    parts.append(f"{drug.drug_name} ({drug.chembl_id}) is a {drug.approval_status} {drug.molecule_type}.")
    
    if drug.synonyms:
        parts.append(f"Also known as: {', '.join(drug.synonyms[:5])}.")
    
    # Mechanisms
    for mech in drug.mechanisms:
        parts.append(
            f"{drug.drug_name} acts as a {mech.action_type} of {mech.target_name} "
            f"(gene: {mech.target_gene_symbol}). "
            f"Mechanism: {mech.mechanism_of_action}."
        )
    
    # Indications
    if drug.indications:
        indication_names = [ind.mesh_heading for ind in drug.indications[:10]]
        parts.append(f"Approved indications include: {', '.join(indication_names)}.")
    
    # Gene summary
    if drug.all_gene_symbols:
        parts.append(f"Target genes: {', '.join(drug.all_gene_symbols)}.")
    
    return " ".join(parts)
```

**Example Output:**
```
LORATADINE (CHEMBL998) is a FDA Approved Small molecule. Also known as: Claritin, 
Claratyne. LORATADINE acts as a ANTAGONIST of Histamine H1 receptor (gene: HRH1). 
Mechanism: Histamine H1 receptor antagonist. Approved indications include: 
Rhinitis, Allergic, Seasonal, Urticaria, Chronic. Target genes: HRH1.
```

### 4.4 Qdrant Payload Schema

```python
{
    # === Identifiers ===
    "doc_id": "uuid-string",
    "chembl_id": "CHEMBL998",
    "drug_name": "LORATADINE",
    "synonyms": ["Claritin", "Claratyne"],
    
    # === Approval ===
    "max_phase": 4,
    "approval_status": "FDA Approved",
    "first_approval": 1993,
    
    # === Mechanisms (Flattened for filtering) ===
    "mechanism_of_action": "Histamine H1 receptor antagonist",
    "action_types": ["ANTAGONIST"],
    "target_gene_symbols": ["HRH1"],           # ← KEY FILTER FIELD
    "target_names": ["Histamine H1 receptor"],
    "target_chembl_ids": ["CHEMBL231"],
    
    # === Indications (Flattened) ===
    "indication_mesh_ids": ["D006255"],
    "indication_names": ["Rhinitis, Allergic, Seasonal"],
    "indication_efo_ids": ["EFO_0003931"],
    
    # === Text ===
    "text_content": "LORATADINE (CHEMBL998) is a FDA Approved...",
    
    # === Metadata ===
    "doc_type": "chembl_drug",
    "data_source": "ChEMBL",
    "created_at": "2026-01-19T12:00:00Z"
}
```

---

## 5. Implementation Phases

### Phase 1: Setup & Configuration (30 minutes)

**Tasks:**
1. Create folder structure ✓
2. Create `chembl_config.yaml`
3. Create data models (`chembl_models.py`)
4. Create `__init__.py` files
5. Add `chembl_webresource_client` to requirements

**Files Created:**
- `chembl/config/chembl_config.yaml`
- `chembl/models/chembl_models.py`
- `chembl/__init__.py` and sub-packages

---

### Phase 2: Data Fetching (2-3 hours)

**Tasks:**
1. Implement `chembl_api_client.py` with retry logic
2. Implement `molecule_fetcher.py` for approved drugs
3. Implement `mechanism_fetcher.py` for mechanisms
4. Implement `indication_fetcher.py` for indications
5. Add caching to avoid redundant API calls

**Key Implementation Details:**

```python
# molecule_fetcher.py - Core logic
from chembl_webresource_client.new_client import new_client

class MoleculeFetcher:
    def __init__(self):
        self.molecule = new_client.molecule
        
    def fetch_approved_drugs(self, batch_size=100):
        """Fetch all approved drugs (max_phase=4)."""
        # Returns ~2,795 drugs
        return self.molecule.filter(max_phase=4)
    
    def fetch_by_chembl_ids(self, chembl_ids: List[str]):
        """Fetch specific molecules by ID."""
        return self.molecule.filter(
            molecule_chembl_id__in=chembl_ids
        )
```

**Expected Output:**
- `cache/molecules.json` - All approved molecules
- `cache/mechanisms.json` - All mechanisms
- `cache/indications.json` - All indications

---

### Phase 3: Data Parsing & Normalization (2 hours)

**Tasks:**
1. Implement `chembl_normalizer.py`
2. Map ChEMBL fields to our schema
3. Extract gene symbols from target components
4. Generate searchable text content
5. Handle missing data gracefully

**Key Transformations:**

| ChEMBL Field | Our Field | Transformation |
|--------------|-----------|----------------|
| `pref_name` | `drug_name` | Direct map |
| `max_phase=4` | `approval_status` | → "FDA Approved" |
| `target_components[].gene_symbol` | `target_gene_symbols` | Flatten list |
| `mesh_heading` | `indication_names` | Collect all |
| Multiple fields | `text_content` | Concatenate |

---

### Phase 4: Embedding & Ingestion (2 hours)

**Tasks:**
1. Create `ChEMBL_drugs` collection in Qdrant
2. Implement `batch_processor.py` for embedding
3. Implement `chembl_ingest.py` orchestrator
4. Batch upsert to Qdrant
5. Implement progress tracking

**Collection Configuration:**
```python
# Same embedding model as existing collection
VECTOR_SIZE = 768  # PubMedBERT

collection_config = {
    "collection_name": "ChEMBL_drugs",
    "vectors_config": {
        "size": VECTOR_SIZE,
        "distance": "Cosine"
    }
}
```

**Estimated Processing:**
- ~2,795 approved drugs
- ~5,000-10,000 vectors (with chunking)
- ~30-60 minutes for full ingestion

---

### Phase 5: Unified Search Integration (2 hours)

**Tasks:**
1. Create `unified_search.py` module
2. Modify `hybrid_search.py` to query both collections
3. Implement result merging with deduplication
4. Implement unified ranking
5. Update retrieval API

**Search Strategy:**
```python
class UnifiedSearch:
    def search(self, query: str, gene_symbols: List[str] = None):
        # 1. Search existing Drug_agent collection
        results_gene_centric = self.search_drug_agent(query, gene_symbols)
        
        # 2. Search new ChEMBL_drugs collection
        results_chembl = self.search_chembl_drugs(query, gene_symbols)
        
        # 3. Merge and deduplicate
        merged = self.merge_results(results_gene_centric, results_chembl)
        
        # 4. Re-rank unified results
        ranked = self.rank_unified(merged)
        
        return ranked
```

---

### Phase 6: Testing & Validation (1 hour)

**Test Cases:**

| Test | Description | Expected |
|------|-------------|----------|
| Gene Lookup | Search for "HRH1" | Returns Loratadine, Cetirizine |
| Drug Lookup | Search for "Imatinib" | Returns BCR-ABL mechanism |
| Disease Lookup | Search for "allergic rhinitis" | Returns antihistamines |
| Cross-collection | Gene from JSON + ChEMBL drug | Both in results |

---

## 6. Module Specifications

### 6.1 chembl_api_client.py

```python
"""
ChEMBL API Client with retry logic and caching.
"""

class ChEMBLAPIClient:
    """Wrapper around chembl_webresource_client with enhancements."""
    
    def __init__(self, cache_dir: str = "./cache", max_retries: int = 3):
        self.cache_dir = Path(cache_dir)
        self.max_retries = max_retries
        self._init_clients()
    
    def _init_clients(self):
        """Initialize ChEMBL API clients."""
        from chembl_webresource_client.new_client import new_client
        self.molecule = new_client.molecule
        self.mechanism = new_client.mechanism
        self.indication = new_client.drug_indication
        self.target = new_client.target
    
    def fetch_with_retry(self, fetch_func, *args, **kwargs):
        """Execute fetch with exponential backoff retry."""
        pass
    
    def cache_response(self, key: str, data: Any):
        """Cache API response to disk."""
        pass
    
    def load_cached(self, key: str) -> Optional[Any]:
        """Load cached response if exists and fresh."""
        pass
```

### 6.2 chembl_normalizer.py

```python
"""
Normalize ChEMBL data to our internal schema.
"""

class ChEMBLNormalizer:
    """Transform raw ChEMBL data to normalized documents."""
    
    def normalize_molecule(self, raw: Dict) -> ChEMBLMolecule:
        """Normalize molecule data."""
        pass
    
    def normalize_mechanism(self, raw: Dict) -> MechanismInfo:
        """Normalize mechanism data, extract gene symbols."""
        pass
    
    def normalize_indication(self, raw: Dict) -> IndicationInfo:
        """Normalize indication data."""
        pass
    
    def create_drug_document(
        self,
        molecule: ChEMBLMolecule,
        mechanisms: List[MechanismInfo],
        indications: List[IndicationInfo]
    ) -> ChEMBLDrugDocument:
        """Combine all data into final document."""
        pass
```

### 6.3 chembl_ingest.py

```python
"""
Main ingestion orchestrator for ChEMBL data.
"""

class ChEMBLIngestion:
    """Orchestrates the full ChEMBL ingestion pipeline."""
    
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.fetcher = ChEMBLAPIClient()
        self.normalizer = ChEMBLNormalizer()
        self.embedder = get_embedder()  # PubMedBERT
        self.storage = QdrantStorage(collection="ChEMBL_drugs")
    
    async def run_full_ingestion(self):
        """Run complete ingestion pipeline."""
        # Phase 1: Fetch
        molecules = await self.fetch_approved_drugs()
        mechanisms = await self.fetch_all_mechanisms(molecules)
        indications = await self.fetch_all_indications(molecules)
        
        # Phase 2: Normalize
        documents = self.normalize_all(molecules, mechanisms, indications)
        
        # Phase 3: Embed & Store
        await self.embed_and_store(documents)
    
    async def fetch_approved_drugs(self) -> List[Dict]:
        """Fetch all approved drugs from ChEMBL."""
        pass
    
    async def embed_and_store(self, documents: List[ChEMBLDrugDocument]):
        """Generate embeddings and store in Qdrant."""
        pass
```

---

## 7. API Reference

### 7.1 ChEMBL API Endpoints Used

#### Molecules (Approved Drugs)
```
GET https://www.ebi.ac.uk/chembl/api/data/molecule?max_phase=4&format=json
```

**Filters:**
- `max_phase=4` - Only approved drugs
- `molecule_type=Small molecule` - Filter by type
- `pref_name__contains=X` - Search by name

#### Mechanisms of Action
```
GET https://www.ebi.ac.uk/chembl/api/data/mechanism?molecule_chembl_id=CHEMBL998&format=json
```

**Key Fields:**
- `mechanism_of_action` - Text description
- `action_type` - ANTAGONIST, INHIBITOR, etc.
- `target_chembl_id` - Link to target

#### Drug Indications
```
GET https://www.ebi.ac.uk/chembl/api/data/drug_indication?molecule_chembl_id=CHEMBL998&format=json
```

**Key Fields:**
- `mesh_heading` - Disease name
- `efo_term` - EFO disease term
- `max_phase_for_ind` - Approval status for indication

#### Targets
```
GET https://www.ebi.ac.uk/chembl/api/data/target?target_chembl_id=CHEMBL231&format=json
```

**Key Fields:**
- `pref_name` - Target name
- `target_type` - SINGLE PROTEIN, PROTEIN COMPLEX, etc.
- `target_components[].gene_symbol` - **Critical for gene linking**

### 7.2 Python Client Examples

```python
from chembl_webresource_client.new_client import new_client

# Initialize clients
molecule = new_client.molecule
mechanism = new_client.mechanism
indication = new_client.drug_indication
target = new_client.target

# Fetch approved drugs
approved = molecule.filter(max_phase=4)
print(f"Found {len(approved)} approved drugs")

# Fetch mechanisms for a drug
mechs = mechanism.filter(molecule_chembl_id='CHEMBL998')
for m in mechs:
    print(f"- {m['mechanism_of_action']}")
    print(f"  Target: {m['target_name']}")
    
# Fetch indications
inds = indication.filter(molecule_chembl_id='CHEMBL998')
for i in inds:
    print(f"- {i['mesh_heading']}")

# Get gene symbol from target
t = target.filter(target_chembl_id='CHEMBL231')[0]
for comp in t.get('target_components', []):
    print(f"Gene: {comp.get('gene_symbol')}")  # HRH1
```

---

## 8. Storage Strategy

### 8.1 Collection Design

**Decision: Separate Collection in Same Instance**

| Aspect | Drug_agent | ChEMBL_drugs |
|--------|------------|--------------|
| Focus | Gene-centric | Drug-centric |
| Source | GeneALaCart JSON | ChEMBL API |
| Key lookup | Gene → Drugs | Drug → Genes |
| Documents | ~50,000+ | ~5,000-10,000 |
| Schema | ParsedGeneData | ChEMBLDrugDocument |

### 8.2 Collection Configuration

```python
# ChEMBL_drugs collection
{
    "collection_name": "ChEMBL_drugs",
    "vectors_config": {
        "size": 768,           # PubMedBERT dimension
        "distance": "Cosine"
    },
    "optimizers_config": {
        "indexing_threshold": 10000
    },
    "on_disk_payload": False   # Keep in memory for speed
}
```

### 8.3 Index Strategy

Create payload indexes for efficient filtering:

```python
# Indexes for ChEMBL_drugs collection
indexes = [
    ("target_gene_symbols", "keyword"),  # Filter by gene
    ("chembl_id", "keyword"),            # Exact lookup
    ("drug_name", "text"),               # Text search
    ("indication_names", "text"),        # Disease search
    ("approval_status", "keyword"),      # Filter by status
    ("max_phase", "integer"),            # Filter by phase
]
```

### 8.4 Query Strategy

```python
# Gene-first query (common case)
def search_by_gene(gene_symbol: str):
    # 1. Vector search with text
    query_text = f"drugs targeting {gene_symbol} gene"
    
    # 2. Add filter for exact gene match
    filter = {
        "must": [
            {"key": "target_gene_symbols", "match": {"value": gene_symbol}}
        ]
    }
    
    # 3. Search ChEMBL_drugs
    results = qdrant.search(
        collection="ChEMBL_drugs",
        query_vector=embed(query_text),
        query_filter=filter,
        limit=20
    )
    
    return results
```

---

## 9. Search & Retrieval

### 9.1 Unified Search Flow

```
User Input: "Find drugs for EGFR in lung cancer"
                    │
                    ▼
            ┌───────────────┐
            │ Query Parser  │
            │ - genes: EGFR │
            │ - disease:    │
            │   lung cancer │
            └───────┬───────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│  Drug_agent   │       │ ChEMBL_drugs  │
│  Collection   │       │  Collection   │
│               │       │               │
│ Vector Search │       │ Vector Search │
│ + Gene Filter │       │ + Gene Filter │
└───────┬───────┘       └───────┬───────┘
        │                       │
        │    ┌──────────┐       │
        └───►│  Merge   │◄──────┘
             │ & Dedup  │
             └────┬─────┘
                  │
                  ▼
           ┌───────────┐
           │ Re-Rank   │
           │ (Unified) │
           └─────┬─────┘
                 │
                 ▼
          ┌────────────┐
          │  Top 15    │
          │  Results   │
          └────────────┘
```

### 9.2 Merging Strategy

```python
def merge_results(
    gene_centric: List[SearchResult],
    drug_centric: List[SearchResult]
) -> List[MergedResult]:
    """Merge results from both collections."""
    
    merged = {}
    
    # Index gene-centric by drug name
    for result in gene_centric:
        drug_name = result.payload.get("drug_name", "").lower()
        if drug_name:
            merged[drug_name] = MergedResult(
                drug_name=drug_name,
                gene_centric_score=result.score,
                gene_centric_data=result.payload,
                chembl_data=None,
                chembl_score=0.0
            )
    
    # Merge ChEMBL results
    for result in drug_centric:
        drug_name = result.payload.get("drug_name", "").lower()
        if drug_name in merged:
            # Enrich existing result
            merged[drug_name].chembl_data = result.payload
            merged[drug_name].chembl_score = result.score
        else:
            # New drug from ChEMBL
            merged[drug_name] = MergedResult(
                drug_name=drug_name,
                gene_centric_score=0.0,
                gene_centric_data=None,
                chembl_data=result.payload,
                chembl_score=result.score
            )
    
    return list(merged.values())
```

### 9.3 Unified Ranking

```python
def rank_unified(merged_results: List[MergedResult]) -> List[MergedResult]:
    """Rank merged results with unified scoring."""
    
    for result in merged_results:
        score = 0.0
        
        # Relevance from both collections (max)
        relevance = max(result.gene_centric_score, result.chembl_score)
        score += 0.30 * relevance
        
        # Gene match bonus (from either source)
        if result.has_gene_match:
            score += 0.35
        
        # Evidence level (prefer ChEMBL with mechanism)
        if result.chembl_data and result.chembl_data.get("mechanism_of_action"):
            score += 0.20
        elif result.gene_centric_data:
            score += 0.10
        
        # Approval status (from ChEMBL)
        if result.chembl_data:
            phase = result.chembl_data.get("max_phase", 0)
            score += 0.15 * (phase / 4.0)
        
        result.unified_score = score
    
    # Sort by unified score
    return sorted(merged_results, key=lambda x: x.unified_score, reverse=True)
```

---

## 10. Testing Plan

### 10.1 Unit Tests

```python
# tests/test_chembl_fetcher.py

def test_fetch_approved_drugs():
    """Verify approved drugs are fetched correctly."""
    fetcher = MoleculeFetcher()
    drugs = fetcher.fetch_approved_drugs()
    
    assert len(drugs) > 2000  # Should have ~2795
    assert all(d['max_phase'] == 4 for d in drugs)

def test_fetch_mechanisms():
    """Verify mechanisms include gene symbols."""
    fetcher = MechanismFetcher()
    mechs = fetcher.fetch_for_molecule('CHEMBL998')
    
    assert len(mechs) > 0
    assert any('HRH1' in str(m) for m in mechs)

def test_normalize_molecule():
    """Verify molecule normalization."""
    normalizer = ChEMBLNormalizer()
    raw = {"pref_name": "LORATADINE", "max_phase": 4}
    
    result = normalizer.normalize_molecule(raw)
    
    assert result.drug_name == "LORATADINE"
    assert result.approval_status == "FDA Approved"
```

### 10.2 Integration Tests

```python
# tests/test_chembl_integration.py

def test_full_ingestion_pipeline():
    """Test complete ingestion of sample drugs."""
    ingestion = ChEMBLIngestion()
    
    # Ingest 10 drugs
    stats = ingestion.run_sample_ingestion(limit=10)
    
    assert stats.documents_ingested == 10
    assert stats.errors == 0

def test_search_by_gene():
    """Test searching ChEMBL by gene symbol."""
    search = UnifiedSearch()
    
    results = search.search_by_gene("HRH1")
    
    assert len(results) > 0
    assert any("LORATADINE" in r.drug_name.upper() for r in results)

def test_unified_search():
    """Test merged search across both collections."""
    search = UnifiedSearch()
    
    results = search.search(
        query="drugs targeting EGFR for lung cancer",
        gene_symbols=["EGFR"]
    )
    
    # Should have results from both collections
    assert any(r.source == "Drug_agent" for r in results)
    assert any(r.source == "ChEMBL_drugs" for r in results)
```

### 10.3 Validation Queries

| Query | Expected Top Results |
|-------|---------------------|
| Gene: HRH1 | Loratadine, Cetirizine, Diphenhydramine |
| Gene: EGFR | Erlotinib, Gefitinib, Osimertinib |
| Gene: BCR-ABL | Imatinib, Dasatinib, Nilotinib |
| Disease: Asthma | Montelukast, Fluticasone |
| Drug: Aspirin | Shows COX-1/COX-2 mechanism |

---

## 11. Troubleshooting

### 11.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| API timeout | Large result set | Use pagination, reduce batch size |
| Missing gene symbols | Target has no components | Skip or use target name |
| Duplicate drugs | Same drug in both collections | Deduplicate by name |
| Empty mechanisms | Drug has no known target | Include with lower rank |
| Rate limiting | Too many requests | Add delays, use caching |

### 11.2 Debugging Commands

```bash
# Test ChEMBL API connection
python -c "from chembl_webresource_client.new_client import new_client; print(len(new_client.molecule.filter(max_phase=4)[:10]))"

# Validate Qdrant collection
python -c "from qdrant_client import QdrantClient; c = QdrantClient(...); print(c.get_collection('ChEMBL_drugs'))"

# Count documents
python -c "from storage.qdrant_client import QdrantStorage; s = QdrantStorage(collection='ChEMBL_drugs'); print(s.count())"
```

### 11.3 Logging

Enable debug logging for troubleshooting:

```python
import logging
logging.getLogger("chembl").setLevel(logging.DEBUG)
logging.getLogger("qdrant_client").setLevel(logging.DEBUG)
```

---

## Appendix A: Dependencies

Add to `requirements.txt`:

```
chembl_webresource_client>=0.10.8
```

## Appendix B: Environment Variables

```bash
# .env additions
CHEMBL_CACHE_DIR=./chembl/cache
CHEMBL_COLLECTION=ChEMBL_drugs
CHEMBL_BATCH_SIZE=100
```

## Appendix C: Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Setup | 30 min | None |
| Phase 2: Fetching | 2-3 hours | ChEMBL API |
| Phase 3: Parsing | 2 hours | Phase 2 |
| Phase 4: Ingestion | 2 hours | Phase 3, Qdrant |
| Phase 5: Search | 2 hours | Phase 4 |
| Phase 6: Testing | 1 hour | Phase 5 |
| **Total** | **9-10 hours** | |

---

**End of Documentation**
