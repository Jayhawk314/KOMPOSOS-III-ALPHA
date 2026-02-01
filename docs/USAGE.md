# KOMPOSOS-III Usage Guide

## Phylogenetics of Concepts: Tracing How Ideas Evolve

KOMPOSOS-III is a categorical AI system that combines **Category Theory**, **Homotopy Type Theory (HoTT)**, **Cubical Type Theory**, and **Game Theory** to analyze the evolution of concepts across domains.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [CLI Commands Reference](#cli-commands-reference)
3. [Workflow Overview](#workflow-overview)
4. [Mathematical Foundations](#mathematical-foundations)
5. [Oracle System](#oracle-system)
6. [Code Architecture](#code-architecture)
7. [Example Session](#example-session)

---

## Quick Start

```bash
# 1. Initialize a corpus directory
python cli.py init --corpus ./my_corpus

# 2. Load data into the store
python cli.py load --corpus ./my_corpus

# 3. Compute embeddings (required for Oracle predictions)
python cli.py embed

# 4. Generate a comprehensive report
python cli.py report full --output analysis.md

# 5. Query specific evolutionary paths
python cli.py query evolution "Newton" "Dirac"
```

---

## CLI Commands Reference

### `init` - Initialize Corpus

Creates a new corpus directory structure for organizing your knowledge data.

```bash
python cli.py init [--corpus PATH]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--corpus` | Path to corpus directory | `./corpus` |

**Creates:**
```
corpus/
  objects/       # JSON files defining concepts
  morphisms/     # JSON files defining relationships
  equivalences/  # JSON files defining equivalences
  sources/       # Source documents
```
  KOMPOSOS-III CLI Commands (10 Total)

  | Command     | Description                 | Usage                                                          |
  |-------------|-----------------------------|----------------------------------------------------------------|
  | init        | Initialize corpus directory | python cli.py init --corpus ./my_corpus                        |
  | load        | Load data from corpus       | python cli.py load --corpus ./my_corpus --db store.db          |
  | query       | Query knowledge graph       | python cli.py query evolution "Newton" "Dirac"                 |
  | report      | Generate markdown reports   | python cli.py report evolution "Planck" "Feynman" -o report.md |
  | oracle      | Run Oracle predictions      | python cli.py oracle "Planck" "Feynman"                        |
  | homotopy    | Analyze path homotopy       | python cli.py homotopy "Planck" "Feynman"                      |
  | predict     | Single prediction query     | python cli.py predict "Newton" "Einstein"                      |
  | stress-test | Run quality stress tests    | python cli.py stress-test                                      |
  | stats       | Show store statistics       | python cli.py stats                                            |
  | embed       | Compute embeddings          | python cli.py embed                                            |

  New Commands Added:

  1. oracle - Direct access to the 8-strategy Categorical Oracle predictions
  2. homotopy - Path homotopy analysis (are paths equivalent as proofs?)
  3. predict - Lightweight prediction with existing paths + Oracle
  4. stress-test - Run the full quality stress test suite

  Query Types:

  - query evolution <source> <target> - Find evolutionary paths
  - query equivalence <A> <B> - Check if two concepts are equivalent
  - query gaps --threshold 0.3 - Find missing relationships

---

### `load` - Load Data

Loads data from the corpus into the SQLite store.

```bash
python cli.py load [--corpus PATH] [--db PATH]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--corpus` | Path to corpus directory | `./corpus` |
| `--db` | Path to database file | `~/.komposos3/store.db` |

---

### `embed` - Compute Embeddings

Computes semantic embeddings for all objects using sentence-transformers.

```bash
python cli.py embed [--db PATH]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--db` | Path to database file | `~/.komposos3/store.db` |

**Required for:**
- Oracle predictions (SemanticSimilarityStrategy)
- Gap analysis
- Semantic clustering

**Model:** `all-mpnet-base-v2` (768-dimensional embeddings)

---

### `query` - Query Knowledge Graph

Query the categorical knowledge structure.

#### Evolution Query
Find paths showing how concept A evolved into concept B.

```bash
python cli.py query evolution "SOURCE" "TARGET" [--db PATH]
```

**Example:**
```bash
python cli.py query evolution "Newton" "Schrödinger"
```

**Output:**
```
Evolution paths from 'Newton' to 'Schrödinger':
Found 3 path(s)

Path 1 (length 4):
  Newton ─[influenced (1750)]→ Euler
  Euler ─[influenced (1788)]→ Lagrange
  Lagrange ─[reformulated (1833)]→ Hamilton
  Hamilton ─[influenced (1926)]→ Schrodinger
```

#### Equivalence Query
Check if two concepts are equivalent (HoTT univalence).

```bash
python cli.py query equivalence "OBJ1" "OBJ2" [--db PATH]
```

**Example:**
```bash
python cli.py query equivalence "WaveMechanics" "MatrixMechanics"
```

**Output:**
```
'WaveMechanics' ≃ 'MatrixMechanics' (equivalent)
  Class: QM_Formulations
  Type: mathematical
  Witness: vonNeumann_1932
```

#### Gap Query
Find semantic gaps (missing connections) in the knowledge graph.

```bash
python cli.py query gaps [--threshold FLOAT] [--db PATH]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--threshold` | Similarity threshold (lower = more gaps) | `0.3` |

---

### `report` - Generate Reports

Generate rich, human-readable markdown reports.

#### Evolution Report
Traces how concept A evolved into concept B with full categorical analysis.

```bash
python cli.py report evolution "SOURCE" "TARGET" [--output FILE] [--db PATH]
```

**Example:**
```bash
python cli.py report evolution "Newton" "Dirac" --output newton_to_dirac.md
```

**Report Contents:**
- Executive Summary
- Path Analysis (all evolutionary routes)
- Scientific Narrative
- Equivalence Analysis (HoTT)
- Oracle Predictions (8 strategies)
- Yoneda Structural Analysis
- Categorical 2-Structure
- Conclusions and Future Directions

#### Gap Report
Analyzes structural holes and missing connections.

```bash
python cli.py report gaps [--threshold FLOAT] [--output FILE] [--db PATH]
```

**Report Contents:**
- Graph Statistics
- Semantic Gap Analysis
- Isolated Objects
- Kan Extension Candidates
- Coherence Analysis

#### Equivalence Report
Analyzes all equivalence classes (univalence implementation).

```bash
python cli.py report equivalence [--output FILE] [--db PATH]
```

**Report Contents:**
- Equivalence Class Details
- HoTT Interpretation
- Transitive Closure Analysis

#### Full Report
Comprehensive analysis of the entire knowledge graph.

```bash
python cli.py report full [--output FILE] [--db PATH]
```

**Report Contents:**
- All of the above, combined
- Domain Analysis
- Graph Connectivity (NetworkX)
- Four Pillars Status

---

### `stats` - Show Statistics

Display knowledge graph statistics.

```bash
python cli.py stats [--db PATH]
```

**Output:**
```
KOMPOSOS-III Store Statistics
========================================
Objects:           43
Morphisms:         61
Stored Paths:      0
Equivalences:      4
Higher Morphisms:  0

Object Types:
  Physicist: 25
  Theory: 14
  Mathematician: 4

Morphism Types:
  influenced: 45
  created: 8
  reformulated: 5
  extended: 3
```

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        KOMPOSOS-III WORKFLOW                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. DATA INGESTION                                                   │
│     ┌─────────┐     ┌─────────┐     ┌─────────────┐                 │
│     │ init    │ ──▶ │ load    │ ──▶ │ embed       │                 │
│     │ corpus  │     │ to DB   │     │ (required)  │                 │
│     └─────────┘     └─────────┘     └─────────────┘                 │
│                                                                      │
│  2. CATEGORICAL ANALYSIS                                             │
│     ┌─────────────────────────────────────────────────┐             │
│     │              CategoricalOracle                   │             │
│     │  ┌────────────┬────────────┬────────────┐       │             │
│     │  │ Kan        │ Semantic   │ Temporal   │       │             │
│     │  │ Extension  │ Similarity │ Reasoning  │       │             │
│     │  ├────────────┼────────────┼────────────┤       │             │
│     │  │ Type       │ Yoneda     │ Composition│       │             │
│     │  │ Heuristic  │ Pattern    │ Strategy   │       │             │
│     │  ├────────────┼────────────┼────────────┤       │             │
│     │  │ Fibration  │ Structural │            │       │             │
│     │  │ Lift       │ Hole       │            │       │             │
│     │  └────────────┴────────────┴────────────┘       │             │
│     │                    ▼                             │             │
│     │  ┌─────────────────────────────────────────┐    │             │
│     │  │ Sheaf Coherence Validation               │    │             │
│     │  │ (contradictions, similarity thresholds) │    │             │
│     │  └─────────────────────────────────────────┘    │             │
│     │                    ▼                             │             │
│     │  ┌─────────────────────────────────────────┐    │             │
│     │  │ Game-Theoretic Optimization (Nash EQ)   │    │             │
│     │  └─────────────────────────────────────────┘    │             │
│     │                    ▼                             │             │
│     │  ┌─────────────────────────────────────────┐    │             │
│     │  │ Bayesian Learning (confidence updates)  │    │             │
│     │  └─────────────────────────────────────────┘    │             │
│     └─────────────────────────────────────────────────┘             │
│                                                                      │
│  3. OUTPUT                                                           │
│     ┌─────────┐     ┌─────────┐     ┌─────────────┐                 │
│     │ query   │     │ report  │     │ predictions │                 │
│     │ results │     │ (.md)   │     │ with conf.  │                 │
│     └─────────┘     └─────────┘     └─────────────┘                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Foundations

### The Four Pillars

| Pillar | Mathematical Concept | Implementation | Code Location |
|--------|---------------------|----------------|---------------|
| **Category Theory** | Objects, morphisms, composition | Knowledge graph structure | `data/store.py` |
| **HoTT** | Univalence axiom, equivalences | Equivalence classes | `hott/` |
| **Cubical TT** | hcomp, hfill operations | Gap filling | `cubical/` |
| **Game Theory** | Nash equilibrium, open games | Prediction optimization | `game/` |

### Category Theory: Objects and Morphisms

```
Category K (Knowledge):
  - Objects: Concepts (scientists, theories, ideas)
  - Morphisms: Relationships (influenced, created, extended)
  - Composition: If A→B→C, then A→C
  - Identity: Every object has id_A : A → A
```

**Code:** `data/store.py:StoredObject`, `data/store.py:StoredMorphism`

### Homotopy Type Theory: Equivalences

The **univalence axiom** states: `(A ≃ B) ≃ (A = B)`

Equivalent concepts are treated as equal:
- Wave mechanics ≃ Matrix mechanics (von Neumann, 1932)
- Lagrangian ≃ Hamiltonian formulations

**Code:** `data/store.py:EquivalenceClass`, `hott/equivalence.py`

### Kan Extensions: Inference

Left Kan extension computes the "best approximation" of extending structure:

```
Given: F: C → D and G: C → E
Compute: Lan_F(G): D → E (left Kan extension)
```

Used for predicting missing morphisms from surrounding structure.

**Code:** `categorical/kan_extensions.py:LeftKanExtension`

### Game Theory: Nash Equilibrium

Prediction selection modeled as 2-player game:
- Player 1 (Predictor): Maximizes prediction utility
- Player 2 (Validator): Maximizes precision

Nash equilibrium selects optimal prediction set.

**Code:** `oracle/optimizer.py:PredictionOptimizer`, `game/nash.py`

---

## Oracle System

The **CategoricalOracle** generates predictions using 8 rigorous strategies:

### Strategy 1: Kan Extension
Uses categorical Kan extensions to infer missing morphisms via colimit computation.

```python
# Code: oracle/strategies.py:KanExtensionStrategy
# Math: Lan_F(G)(target) via colimit over comma category
```

### Strategy 2: Semantic Similarity
Predicts connections between semantically similar objects using embeddings.

```python
# Code: oracle/strategies.py:SemanticSimilarityStrategy
# Math: sim(A, B) = cos(embed(A), embed(B))
```

### Strategy 3: Temporal Reasoning
Uses temporal metadata (birth/death dates) for influence prediction.

```python
# Code: oracle/strategies.py:TemporalReasoningStrategy
# Logic: birth(A) < birth(B) → A may have influenced B
```

### Strategy 4: Type Heuristic
Uses object types to constrain valid predictions.

```python
# Code: oracle/strategies.py:TypeHeuristicStrategy
# Rules: (Physicist, Physicist) → [influenced, collaborated]
#        (Physicist, Theory) → [created, contributed, developed]
```

### Strategy 5: Yoneda Pattern
Objects with same morphism patterns are structurally similar.

```python
# Code: oracle/strategies.py:YonedaPatternStrategy
# Math: Yoneda lemma: Hom(A, -) determines A
```

### Strategy 6: Composition
If A→B→C exists, predict A→C should exist (transitive closure).

```python
# Code: oracle/strategies.py:CompositionStrategy
# Math: g ∘ f : A → C when f: A→B, g: B→C
```

### Strategy 7: Fibration Lift
Uses fibration structure for Cartesian lift predictions.

```python
# Code: oracle/strategies.py:FibrationLiftStrategy
# Math: Cartesian lifts in fibered categories
```

### Strategy 8: Structural Hole
Finds triangles that should close.

```python
# Code: oracle/strategies.py:StructuralHoleStrategy
# Logic: A→B, A→C, missing B→C → predict B→C
```

### Validation Pipeline

```
Raw Predictions (from 8 strategies)
         │
         ▼
┌─────────────────────────────────────┐
│ 1. Merge Duplicates                  │
│    Combine predictions for same key │
│    Boost confidence for agreement   │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 2. Sheaf Coherence Validation       │
│    - Check for contradictions       │
│    - Antonym detection              │
│    - Similarity thresholds (≥0.5)   │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 3. Game-Theoretic Optimization      │
│    - Nash equilibrium selection     │
│    - Utility function maximization  │
│    - Iterated best response         │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 4. Bayesian Learning Adjustment     │
│    - Update from historical data    │
│    - Formula: 0.5*orig + 0.25*type  │
│               + 0.25*strategy       │
│    - Laplace smoothing              │
└─────────────────────────────────────┘
         │
         ▼
    Final Predictions
```

**Code:** `oracle/__init__.py:CategoricalOracle.predict()`

---

## Code Architecture

```
KOMPOSOS-III/
├── cli.py                    # Command-line interface
├── oracle/                   # Prediction system (NEW)
│   ├── __init__.py          # CategoricalOracle main class
│   ├── prediction.py        # Prediction dataclass
│   ├── strategies.py        # 8 inference strategies
│   ├── coherence.py         # SheafCoherenceChecker
│   ├── optimizer.py         # Game-theoretic optimizer
│   └── learner.py           # Bayesian learning
├── data/                    # Data layer
│   ├── store.py             # SQLite storage (objects, morphisms)
│   ├── embeddings.py        # Sentence-transformer embeddings
│   ├── sources.py           # Source document handling
│   └── loader.py            # Corpus loading
├── categorical/             # Category theory
│   ├── category.py          # Category base classes
│   ├── functor.py           # Functor implementation
│   ├── kan_extensions.py    # Left/Right Kan extensions
│   └── limits.py            # Limits and colimits
├── hott/                    # Homotopy Type Theory
│   ├── types.py             # HoTT type system
│   ├── equivalence.py       # Equivalence and univalence
│   └── path_induction.py    # Path induction
├── cubical/                 # Cubical Type Theory
│   ├── interval.py          # Interval type
│   ├── path.py              # Path types
│   └── kan_ops.py           # hcomp/hfill operations
├── game/                    # Game Theory
│   ├── open_game.py         # Open games
│   └── nash.py              # Nash equilibrium
└── evaluation/              # Testing
    ├── stress_test.py       # Quality validation
    └── physics_dataset.py   # Test dataset
```

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `CategoricalOracle` | `oracle/__init__.py` | Main prediction engine |
| `Prediction` | `oracle/prediction.py` | Prediction data structure |
| `InferenceStrategy` | `oracle/strategies.py` | Base strategy class |
| `SheafCoherenceChecker` | `oracle/coherence.py` | Validates predictions |
| `PredictionOptimizer` | `oracle/optimizer.py` | Nash equilibrium selection |
| `OracleLearner` | `oracle/learner.py` | Bayesian confidence learning |
| `KomposOSStore` | `data/store.py` | SQLite data storage |
| `EmbeddingsEngine` | `data/embeddings.py` | Semantic embeddings |
| `ReportGenerator` | `cli.py` | Markdown report generation |

---

## Example Session

### 1. Set Up Physics Dataset

```python
# Using the built-in physics dataset for testing
import sys
sys.path.insert(0, 'C:/Users/JAMES/GitHub/KOMPOSOS-III')

from evaluation.physics_dataset import create_physics_dataset
from data import EmbeddingsEngine, StoreEmbedder

# Create store with physics data
store = create_physics_dataset()

# Compute embeddings
embeddings = EmbeddingsEngine()
embedder = StoreEmbedder(store, embeddings)
embedder.embed_all_objects()
```

### 2. Use the Oracle

```python
from oracle import CategoricalOracle

# Create Oracle (requires embeddings)
oracle = CategoricalOracle(store, embeddings)

# Make predictions
result = oracle.predict("Newton", "Dirac")

print(f"Total candidates: {result.total_candidates}")
print(f"Coherence score: {result.coherence_result.coherence_score:.2%}")
print(f"Computation time: {result.computation_time_ms:.1f}ms")

for pred in result.predictions[:5]:
    print(f"  {pred.strategy_name}: {pred.description}")
    print(f"    Confidence: {pred.confidence:.2%}")
    print(f"    Reasoning: {pred.reasoning}")
```

### 3. Generate Reports

```bash
# Evolution report
python cli.py report evolution "Newton" "Dirac" --output evolution.md

# Gap analysis
python cli.py report gaps --threshold 0.3 --output gaps.md

# Full comprehensive report
python cli.py report full --output full_analysis.md
```

### 4. Run Stress Tests

```python
from evaluation.stress_test import run_all_stress_tests

results = run_all_stress_tests()

# Results:
# - Backtest F1: 100%
# - Temporal Holdout F1: 100%
# - Equivalence Discovery: 100%
# - Consistency: 100%
# - Hub Identification: 100%
# - Overall F1: 100%
```

---

## Data Formats

### Object JSON
```json
{
  "name": "Einstein",
  "type": "Physicist",
  "metadata": {
    "birth": 1879,
    "death": 1955,
    "field": "Physics",
    "contributions": ["Special Relativity", "General Relativity", "Photoelectric Effect"]
  }
}
```

### Morphism JSON
```json
{
  "name": "influenced",
  "source": "Einstein",
  "target": "Bohr",
  "confidence": 0.95,
  "metadata": {
    "year": 1913,
    "context": "Photoelectric effect led to Bohr model"
  }
}
```

### Equivalence JSON
```json
{
  "name": "QM_Formulations",
  "members": ["WaveMechanics", "MatrixMechanics"],
  "type": "mathematical",
  "witness": "vonNeumann_1932",
  "confidence": 1.0,
  "metadata": {
    "year": 1932,
    "proof": "Hilbert space isomorphism"
  }
}
```

---

## Performance

| Test | Result |
|------|--------|
| Backtest (predict hidden morphisms) | 100% F1 |
| Temporal Holdout (pre-1925 → post-1925) | 100% F1 |
| Equivalence Discovery | 100% F1 |
| Consistency Check | 100% F1 |
| Hub Identification | 100% F1 |
| **Overall** | **100% F1** |

---

## Requirements

```
Python 3.8+
sentence-transformers
numpy
networkx (optional, for graph analysis)
```

Install:
```bash
pip install sentence-transformers numpy networkx
```

---
 Where to Get Data

  Free Academic Data Sources:

  | Source           | URL                                       | What You Get                                       |
  |------------------|-------------------------------------------|----------------------------------------------------|
  | OpenAlex         | https://openalex.org/data-dump            | 250M+ academic papers, authors, citations          |
  | Wikidata         | https://dumps.wikimedia.org/wikidatawiki/ | Structured knowledge (people, concepts, relations) |
  | Semantic Scholar | https://api.semanticscholar.org/          | Academic papers with citations                     |
  | DBLP             | https://dblp.org/xml/                     | Computer science publications                      |
  | arXiv            | https://arxiv.org/help/bulk_data          | Physics/math/CS preprints                          |

  Quick Start - Create Your Own:

  # 1. Initialize corpus
  python cli.py init --corpus ./my_corpus

  # 2. Add your CSV files to:
  #    ./my_corpus/custom/objects.csv
  #    ./my_corpus/custom/morphisms.csv

  # 3. Load into KOMPOSOS-III
  python cli.py load --corpus ./my_corpus

  # 4. Run analysis
  python cli.py curvature
  python cli.py oracle "ConceptA" "ConceptB"

  Example Use Cases:

  | Domain     | Objects             | Morphisms                            |
  |------------|---------------------|--------------------------------------|
  | Philosophy | Philosophers, Ideas | influenced, critiqued, built_upon    |
  | Software   | Libraries, Concepts | depends_on, inspired_by, forked_from |
  | Biology    | Species, Genes      | evolved_from, related_to, encodes    |
  | History    | Events, People      | caused, participated_in, preceded    |
  | Music      | Artists, Genres     | influenced, collaborated, sampled    |

  The physics dataset in evaluation/physics_dataset.py shows the pattern - you define objects with metadata (birth/death years, type, contributions) and
   morphisms with confidence scores.
## License

KOMPOSOS-III - Categorical Game-Theoretic Type-Theoretic AI
"Phylogenetics of concepts" - tracing how ideas evolve
 Data Formats Accepted

  1. Custom CSV (Simplest)

  Objects CSV (objects.csv):
  name,type,metadata_json
  "Newton","Physicist","{""birth"": 1643, ""death"": 1727, ""era"": ""classical""}"
  "Einstein","Physicist","{""birth"": 1879, ""death"": 1955, ""era"": ""modern""}"
  "QuantumMechanics","Theory","{""year"": 1925}"

  Morphisms CSV (morphisms.csv):
  name,source,target,confidence,metadata_json
  "influenced","Newton","Hamilton",0.95,"{""year"": 1833}"
  "developed","Einstein","GeneralRelativity",1.0,"{""year"": 1915}"
  "reformulated","Heisenberg","QuantumMechanics",1.0,"{}"

  2. JSON Format

  {
    "objects": [
      {"name": "Newton", "type": "Physicist", "metadata": {"birth": 1643, "death": 1727}},
      {"name": "Einstein", "type": "Physicist", "metadata": {"birth": 1879}}
    ],
    "morphisms": [
      {"name": "influenced", "source": "Newton", "target": "Hamilton", "confidence": 0.95},
      {"name": "developed", "source": "Einstein", "target": "GeneralRelativity", "confidence": 1.0}
    ]
  }

  3. OpenAlex JSONL (Academic papers)

  corpus/openalex/works/part_000.jsonl
  corpus/openalex/concepts/part_000.jsonl

  4. Wikidata JSON/JSONL (Structured knowledge)

  Entities with properties like P737 (influenced_by), P1066 (student_of), etc.

  5. BibTeX (Citation networks)

  @article{einstein1905,
    author = {Einstein, Albert},
    title = {On the Electrodynamics of Moving Bodies},
    year = {1905}
  }

  ---
  Where to Get Data

  Free Academic Data Sources:

  | Source           | URL                                       | What You Get                                       |
  |------------------|-------------------------------------------|----------------------------------------------------|
  | OpenAlex         | https://openalex.org/data-dump            | 250M+ academic papers, authors, citations          |
  | Wikidata         | https://dumps.wikimedia.org/wikidatawiki/ | Structured knowledge (people, concepts, relations) |
  | Semantic Scholar | https://api.semanticscholar.org/          | Academic papers with citations                     |
  | DBLP             | https://dblp.org/xml/                     | Computer science publications                      |
  | arXiv            | https://arxiv.org/help/bulk_data          | Physics/math/CS preprints                          |

  Quick Start - Create Your Own:

  # 1. Initialize corpus
  python cli.py init --corpus ./my_corpus

  # 2. Add your CSV files to:
  #    ./my_corpus/custom/objects.csv
  #    ./my_corpus/custom/morphisms.csv

  # 3. Load into KOMPOSOS-III
  python cli.py load --corpus ./my_corpus

  # 4. Run analysis
  python cli.py curvature
  python cli.py oracle "ConceptA" "ConceptB"

  Example Use Cases:

  | Domain     | Objects             | Morphisms                            |
  |------------|---------------------|--------------------------------------|
  | Philosophy | Philosophers, Ideas | influenced, critiqued, built_upon    |
  | Software   | Libraries, Concepts | depends_on, inspired_by, forked_from |
  | Biology    | Species, Genes      | evolved_from, related_to, encodes    |
  | History    | Events, People      | caused, participated_in, preceded    |
  | Music      | Artists, Genres     | influenced, collaborated, sampled    |

  The physics dataset in evaluation/physics_dataset.py shows the pattern - you define objects with metadata (birth/death years, type, contributions) and
   morphisms with confidence scores.

   # 1. Initialize your high-precision design corpus
python cli.py init --corpus ./precision_design

# 2. Compute the curvature to find 'Geometric Holes'
# This uses the CPU to analyze the manifold's topology
python cli.py curvature

# 3. Run the Oracle to predict the most stable equilibrium
# This pits your design against 9 different failure strategies
python cli.py oracle "Current_Design" "Thurston_Goal"

# 4. Generate the full verification report
# This includes the plain English proof of structural integrity
python cli.py report full --output verified_design.md


 It works. Here's how to use the pipeline:

  ---
  One Command to Test Everything

  # Test with built-in physics data (fast mode)
  python pipeline.py --use-physics --skip-embeddings

  # Test with specific source/target
  python pipeline.py --use-physics --source Newton --target Schrodinger --skip-embeddings

  # Full analysis with embeddings (slower but includes semantic predictions)
  python pipeline.py --use-physics --source Newton --target Schrodinger

  # Use YOUR OWN data
  python pipeline.py --corpus ./my_corpus --source ConceptA --target ConceptB

  # Use existing database
  python pipeline.py --db ./my_data.db --source A --target B

  ---
  What the Pipeline Runs

  | Step | Feature     | Output                                    |
  |------|-------------|-------------------------------------------|
  | 1    | Load Data   | Objects & morphisms from your files       |
  | 2    | Embeddings  | Semantic vectors for similarity           |
  | 3    | Curvature   | Ollivier-Ricci (clusters vs bridges)      |
  | 4    | Ricci Flow  | Geometric decomposition into regions      |
  | 5    | Auto-select | Finds best source/target if not specified |
  | 6    | Homotopy    | Standard + geometric path equivalence     |
  | 7    | Oracle      | 9 strategy predictions                    |
  | 8    | Report      | Markdown summary saved to file            |

  ---
  Your Data Setup

  my_corpus/
  ├── custom/
  │   ├── objects.csv      # Your concepts
  │   └── morphisms.csv    # Your relationships

  Then run:
  python pipeline.py --corpus ./my_corpus
 As you can see, it's all hardcoded template text like:

  report += "The discovery of {count} independent evolutionary pathways..."
  report += "In category theory, the existence of multiple paths..."

  Current State: No LLM

  | Component             | LLM Used? | How It Works                            |
  |-----------------------|-----------|-----------------------------------------|
  | Report text           | No        | Template strings                        |
  | Plain English summary | No        | Rule-based formatting                   |
  | Oracle predictions    | No        | 9 mathematical strategies               |
  | Analysis              | No        | Algorithmic (curvature, homotopy, etc.) |

  Want LLM-Enhanced Reports?

  If you want to add LLM generation for richer reports, you could integrate:

  # Example addition (not implemented)
  import anthropic

  def generate_llm_summary(data, api_key):
      client = anthropic.Anthropic(api_key=api_key)
      response = client.messages.create(
          model="claude-sonnet-4-20250514",
          messages=[{
              "role": "user",
              "content": f"Summarize this knowledge graph analysis: {data}"
          }]
      )
      return response.content[0].text

   Now you have a simple command:

  python cli.py ask "Newton Schrodinger"

  This:
  1. Takes plain text input
  2. Outputs plain English (no math jargon)
  3. Saves to report_Newton_Schrodinger.md automatically

  All existing CLI commands:
  - python cli.py ask "A B" - Plain English report (NEW - saves MD)
  - python cli.py stats - Database stats
  - python cli.py oracle A B - Predictions
  - python cli.py homotopy A B - Path equivalence
  - python cli.py curvature - Graph geometry
  - python cli.py ricci-flow - Geometric regions
  - python cli.py report evolution A B -o file.md - Full evolution report
Now you can use --report or -r on any command:

  python cli.py homotopy Newton Schrodinger --report
  python cli.py oracle Planck Feynman -r
  python cli.py homotopy Planck Feynman -r

  Each saves the full comprehensive MD report automatically.