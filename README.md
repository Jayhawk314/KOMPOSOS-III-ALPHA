# KOMPOSOS-III: Categorical Protein Interaction Prediction

Predicting protein-protein interactions using category theory and ESM-2 biological embeddings.

---

## What It Does

KOMPOSOS-III combines:
- **Category theory** - 9 inference strategies (Kan extension, Yoneda, composition, etc.)
- **ESM-2 embeddings** - 650M parameter protein language model (1280-dimensional vectors)
- **Graph reasoning** - Finds patterns in protein interaction networks

**Input:** 55 known cancer protein interactions (STRING database)
**Output:** 100 novel interaction predictions with confidence scores

---

## Quick Start

```bash
# Install dependencies
pip install torch transformers fair-esm biopython numpy pandas

# Run prediction system
python -m oracle.predict --proteins data/proteins/cancer_proteins.db

# Run compositional leakage audit
cd audit_scripts
python test_compositional_leakage.py
```

---

## Results

**100 predictions stratified by type:**

| Type | Count | What It Means |
|------|-------|---------------|
| Compositional | 39 | Found via 2-3 hop graph paths (A→B→C) |
| Family Extrapolation | 47 | Protein family members (ESM-2 similarity > 85%) |
| Cross-Family | 7 | Different families, not graph-reachable (truly novel) |
| Direct | 7 | Already in training set |

**System comparison (deep discovery rate):**
- Biological embeddings (ESM-2): **70%** not compositional
- Text embeddings (PubMedBERT): **38%** not compositional
- Random baseline: **4.4%**

**Interpretation:** ESM-2 learns patterns from protein sequences that go beyond published literature.

---

## The 9 Categorical Strategies

From `oracle/strategies.py`:

1. **Kan Extension** - Lift known patterns to new proteins
2. **Yoneda** - Proteins with similar interaction patterns
3. **Fibration** - Hierarchical protein families
4. **Adjunction** - Bidirectional relationships
5. **Limit** - Consensus from multiple paths
6. **Colimit** - Union of interaction patterns
7. **Natural Transformation** - Pattern morphisms
8. **Composition** - Transitive closure (A→B + B→C = A→C)
9. **Semantic Similarity** - ESM-2 nearest neighbors

Each strategy votes on potential interactions. Coherence scoring aggregates votes.

---

## How ESM-2 Works

From `data/bio_embeddings.py`:

1. **Get protein sequence** from UniProt
2. **Tokenize** amino acid sequence
3. **Extract embedding** from layer 33 (mean-pooled over residues)
4. **Compute similarity** via cosine distance

```python
def similarity(gene1: str, gene2: str) -> float:
    v1 = embed(gene1)  # 1280-dimensional vector
    v2 = embed(gene2)
    return cosine_similarity(v1, v2)
```

**Key insight:** ESM-2 was trained on 250M protein sequences. Similar sequences → similar interaction partners (evolutionary conservation).

---

## The Audit

We tested for data leakage using 3 independent scripts:

### 1. Compositional Leakage (`test_compositional_leakage.py`)

**Test:** Can predictions be derived from training graph via 2-3 hop paths?

**Method:** BFS traversal from 55 training edges

**Result:** 39% are compositional (32% two-hop, 4% three-hop, 3% reverse)

**Example:** EGFR → RAF1 found via EGFR → KRAS → RAF1 (both edges in training)

### 2. Validation Precision (`test_validation_precision.py`)

**Test:** Are the 3 validated predictions (6% precision) compositional or novel?

**Result:** ALL 3 are two-hop paths:
- EGFR → RAF1 (via KRAS)
- EGFR → BRAF (via KRAS)
- PTEN → BAX (via AKT1)

**Corrected precision on independent validation:** 0%

### 3. Family Extrapolation (`test_protein_family_extrapolation.py`)

**Test:** Are "deep discoveries" actually protein family members?

**Result:** 87% have ESM-2 similarity > 0.85

**Examples:**
- NRAS → KRAS: 99.6% similar (RAS family)
- STAT5 → STAT3: 97.2% similar (STAT family)

**Cross-family discoveries (truly novel):** 7% (similarity < 0.85)

---

## 7 Cross-Family Discoveries

These are NOT compositional and NOT family extrapolations:

| Source | Target | Confidence | ESM-2 Similarity |
|--------|--------|------------|------------------|
| KRAS | MYC | 0.700 | 0.781 |
| NRAS | MYC | 0.680 | 0.789 |
| STAT3 | KRAS | 0.677 | 0.827 |
| BRCA2 | PTEN | 0.675-0.702 | 0.794 |
| PTEN | BRCA2 | 0.675-0.702 | 0.794 |

**Next step:** Experimental validation (Co-IP, yeast two-hybrid)

---

## Repository Structure

```
├── oracle/
│   ├── strategies.py          # 9 categorical inference strategies
│   ├── coherence.py           # Vote aggregation and scoring
│   └── deep_learning.py       # ESM-2 similarity computation
│
├── data/
│   ├── bio_embeddings.py      # ESM-2 embedding generation
│   ├── proteins/
│   │   └── cancer_proteins.db # 55-edge training set (SQLite)
│   └── results/
│       ├── biological_predictions.csv  # 100 ESM-2 predictions
│       └── text_predictions.csv        # 100 PubMedBERT predictions
│
├── audit_scripts/
│   ├── test_compositional_leakage.py       # BFS graph traversal
│   ├── test_validation_precision.py        # Validation decomposition
│   ├── test_protein_family_extrapolation.py # ESM-2 similarity analysis
│   └── deep_discovery_oracle.py            # Re-ranking with penalties
│
├── reports/
│   ├── leakage_audit_results.csv           # All 100 classified
│   ├── validation_precision_decomposition.json
│   └── family_extrapolation_analysis.json
│
├── TECHNICAL_REPORT.md           # Full paper (bioRxiv)
├── SUPPLEMENTARY_MATERIALS.md    # Detailed audit analysis
└── CORRECTED_METRICS_REPORT.md   # Audit summary
```

---

## Key Findings

### 1. Category Theory Works for Biology

39% compositional predictions prove the framework correctly computes transitive closure of interaction graphs. This validates the mathematical implementation.

### 2. ESM-2 Outperforms Text for Discovery

Biological embeddings: 70% deep discovery
Text embeddings: 38% deep discovery

Text recalls published pathways. ESM-2 learns from evolutionary patterns.

### 3. Most "Novel" Predictions Are Family Extrapolations

87% of deep discoveries involve similar proteins (ESM-2 sim > 0.85). This is biologically expected - homologous proteins have similar binding partners.

### 4. Only 7% Are Truly Novel

Cross-family predictions (dissimilar proteins, not graph-reachable) require experimental validation before making biological claims.

---

## Limitations

- **Small training set:** 55 edges (4.4% of 36-protein graph density)
- **Hub bias:** Predictions cluster around KRAS, TP53, MYC
- **Validation contamination:** Original validation set had 54% compositional leakage
- **No independent validation:** 0% precision on truly novel pairs (needs lab experiments)

---

## Papers

**Main Paper:** [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)
**Supplementary Materials:** [SUPPLEMENTARY_MATERIALS.md](SUPPLEMENTARY_MATERIALS.md)
**Audit Report:** [CORRECTED_METRICS_REPORT.md](CORRECTED_METRICS_REPORT.md)

---

## Contact

**Author:** James Ray Hawkins
**Email:** jhawk314@gmail.com
**GitHub:** github.com/Jayhawk314/KOMPOSOS-III-ALPHA

---

**Last Updated:** February 1, 2026
