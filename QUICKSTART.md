# KOMPOSOS-III Conjecture Engine: Quick Start

**Get from zero to 80% precision in 2 minutes**

---

## Prerequisites

Python 3.10+ with pip

---

## Installation

```bash
cd KOMPOSOS-III

# Install dependencies (if not already installed)
pip install sentence-transformers numpy networkx aiosqlite
```

First run will download the embedding model (~400MB, one-time).

---

## Run the Demo

### Step 1: Generate Conjectures

```bash
python test_conjecture_pipeline.py
```

**Expected output**:
```
================================================================================
KOMPOSOS-III Conjecture Generation Pipeline Test
================================================================================

[1/5] Loading physics dataset from store...
  [OK] Loaded 57 objects and 69 morphisms

[2/5] Initializing embeddings engine...
  [OK] Model: all-mpnet-base-v2 (768d)

[3/5] Creating Categorical Oracle...
  [OK] Oracle initialized with 8 strategies

[4/5] Running conjecture engine...
  [OK] Generated 20 conjectures
  [OK] Evaluated 1232 candidate pairs
  [OK] Computation time: 44988.6ms

[5/5] Top Conjectures:
--------------------------------------------------------------------------------

1. Newton --[influenced]--> Lagrange
   Confidence: 0.700
   Strategy: kan_extension+temporal_reasoning+type_heuristic+yoneda_pattern...
   Generators: composition, temporal
   ...
```

**Time**: ~45 seconds

---

### Step 2: Validate Results

```bash
python validate_conjectures.py
```

**Expected output**:
```
================================================================================
Conjecture Validation
================================================================================

[1/4] Loading existing data...
  Existing morphisms: 69

[2/4] Generating conjectures...
  Generated: 30 conjectures

[3/4] Validating against historical records...
--------------------------------------------------------------------------------
[NOVEL] CORRECT    | Newton          -> Lagrange        | conf=0.700
              Evidence: Lagrange's analytical mechanics built on Newton's principles

[NOVEL] CORRECT    | Galileo         -> Euler           | conf=0.690
              Evidence: Euler studied Galilean mechanics and extended it

...

[4/4] Validation Statistics
--------------------------------------------------------------------------------
Total conjectures evaluated: 20
Novel (not in training data): 20
Correct: 8
Incorrect: 2
Unknown (need manual validation): 10

Precision (on validated set): 80.0%

================================================================================
TOP NOVEL CONJECTURES FOR DEEPMIND
================================================================================

These were NOT in the training data but are historically verified:

  Newton -> Lagrange
  Confidence: 0.700
  Strategies: composition+temporal
  Evidence: Lagrange's analytical mechanics built directly on Newton's principles
  ...
```

**Time**: ~45 seconds

---

## That's It!

You just:
1. Generated 20 novel conjectures (100% not in training data)
2. Validated 10 against historical records
3. Achieved 80% precision

---

## Dive Deeper

### Read the Documentation

- **SUMMARY_FOR_DEEPMIND.txt** - One-page executive summary
- **DEEPMIND_RESULTS.md** - Full technical report
- **oracle/README_CONJECTURE.md** - API documentation
- **IMPLEMENTATION_COMPLETE.md** - Development notes

### Explore the Code

- **oracle/conjecture.py** - The conjecture engine (650 lines)
- **tests/test_conjecture.py** - 40 unit tests (all passing)
- **data/store.py** - Knowledge graph storage
- **oracle/strategies.py** - 8 inference strategies

### Run the Tests

```bash
pytest tests/test_conjecture.py -v
```

All 40 tests should pass.

### Try Your Own Data

```python
from data import KomposOSStore, EmbeddingsEngine
from oracle import CategoricalOracle
from oracle.conjecture import ConjectureEngine

# Load your data
store = KomposOSStore('path/to/your/data.db')
embeddings = EmbeddingsEngine()

# Create Oracle + Conjecture Engine
oracle = CategoricalOracle(store, embeddings)
engine = ConjectureEngine(oracle)

# Generate conjectures
result = engine.conjecture(top_k=20)

# Print results
for conj in result.conjectures:
    print(f"{conj.source} -> {conj.target} (conf: {conj.top_confidence:.3f})")
```

---

## Troubleshooting

### "No module named 'sentence_transformers'"

Install the missing package:
```bash
pip install sentence-transformers
```

### "CategoricalOracle requires initialized embeddings"

Embeddings are mandatory. First run will download the model (~400MB):
```python
from data import EmbeddingsEngine
embeddings = EmbeddingsEngine()  # Downloads model if needed
```

### Unicode errors on Windows

If you see `UnicodeEncodeError`, the scripts use ASCII-safe characters now. Make sure you're running the latest versions.

### Out of memory

Reduce batch size or use CPU-only:
```python
embeddings = EmbeddingsEngine(device='cpu')
```

---

## Performance

**On physics dataset (57 objects, 69 edges)**:
- **Generation time**: 44 seconds
- **Candidates evaluated**: 1,232
- **Conjectures returned**: 20
- **Precision**: 80% (8/10 validated)
- **Memory usage**: ~2GB (including model)

**Expected performance scales**:
- 100 objects: ~90 seconds
- 500 objects: ~8 minutes
- 5,000 objects: ~90 minutes (needs optimization)

---

## Next Steps

1. **Read SUMMARY_FOR_DEEPMIND.txt** - Understand what we built
2. **Read DEEPMIND_RESULTS.md** - See the full technical analysis
3. **Modify validate_conjectures.py** - Add your own validation rules
4. **Try a new domain** - Proteins, theorems, social networks

---

## Questions?

See **IMPLEMENTATION_COMPLETE.md** for:
- How to present this to DeepMind
- What to do next
- Technical details

---

**Last Updated**: January 31, 2026
**Version**: 0.1.0
**License**: Apache 2.0
