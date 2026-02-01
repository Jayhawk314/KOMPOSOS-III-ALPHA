# Conjecture Engine

**Flips KOMPOSOS-III from reactive to proactive**

## Overview

The Conjecture Engine transforms the Oracle from answering "Does X relate to Y?" into discovering "Which relationships are missing from the knowledge graph?"

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ConjectureEngine                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ├─→ [1] Generate Candidates (6 strategies)
                   │      • CompositionCandidates      (84 pairs)
                   │      • StructuralHoleCandidates   (92 pairs)
                   │      • FiberCandidates            (371 pairs)
                   │      • SemanticCandidates         (13 pairs)
                   │      • TemporalCandidates         (995 pairs)
                   │      • YonedaCandidates           (57 pairs)
                   │      └─→ Union & Dedupe → 1,232 candidates
                   │
                   ├─→ [2] Score via Oracle.predict()
                   │      Each candidate → full 8-strategy pipeline
                   │      • KanExtensionStrategy
                   │      • SemanticSimilarityStrategy
                   │      • TemporalReasoningStrategy
                   │      • TypeHeuristicStrategy
                   │      • YonedaPatternStrategy
                   │      • CompositionStrategy
                   │      • FibrationLiftStrategy
                   │      • StructuralHoleStrategy
                   │
                   ├─→ [3] Sheaf Coherence Check
                   │      Filter contradictions, boost agreements
                   │
                   ├─→ [4] Game-Theoretic Optimization
                   │      Nash equilibrium selection
                   │
                   └─→ [5] Output: Top-k ranked conjectures
```

## Usage

### Basic

```python
from data import KomposOSStore, EmbeddingsEngine
from oracle import CategoricalOracle
from oracle.conjecture import ConjectureEngine

# Load your data
store = KomposOSStore('path/to/store.db')
embeddings = EmbeddingsEngine()

# Create Oracle (reactive)
oracle = CategoricalOracle(store, embeddings)

# Wrap with Conjecture Engine (proactive)
engine = ConjectureEngine(oracle)

# Generate conjectures
result = engine.conjecture(top_k=20, min_confidence=0.6)

# Examine results
for conj in result.conjectures:
    print(f"{conj.source} -> {conj.target}")
    print(f"  Confidence: {conj.top_confidence:.3f}")
    print(f"  Evidence: {conj.best.reasoning}")
```

### Advanced: Selective Generators

```python
# Only use composition and temporal reasoning
result = engine.conjecture(
    top_k=50,
    min_confidence=0.7,
    generators=["composition", "temporal"]
)
```

### Validation

```python
# After external validation
for conj in result.conjectures:
    pred = conj.best
    was_correct = validate_externally(conj.source, conj.target)
    oracle.record_outcome(pred, was_correct)

# System learns from feedback
print(oracle.get_learning_stats())
```

## Generator Details

### 1. CompositionCandidates

**Insight**: Transitive closure gaps

If A→B and B→C exist but A→C doesn't, propose A→C.

**Example**: Newton→Euler→Lagrange exists, Newton→Lagrange missing → propose it

**Complexity**: O(E²) worst case, but only expands existing edges

### 2. StructuralHoleCandidates

**Insight**: Open triangles

If X→A and X→B exist but A→B doesn't, propose A→B.

**Example**: Einstein→Bohr and Einstein→Heisenberg, but Bohr→Heisenberg missing

**Complexity**: O(E·D²) where D = max degree, capped at 25 siblings

### 3. FiberCandidates

**Insight**: Same (type, era) pairs

Objects in the same category fiber (same type + metadata) likely interact.

**Example**: Two physicists born in same decade, same subfield, no edge

**Complexity**: O(F·N_f²) where F = # fibers, N_f = fiber size (capped at 30)

### 4. SemanticCandidates

**Insight**: Embedding nearest neighbors

High embedding similarity suggests missing relationship.

**Example**: "quantum mechanics" and "wave function" are similar (0.84) but no edge

**Complexity**: O(N·k·log(N)) with k = top_k neighbors

**Requirements**: Embeddings must be precomputed

### 5. TemporalCandidates

**Insight**: Chronological ordering + type compatibility

Older → younger influences, contemporaries collaborate.

**Example**: Galileo (1564) likely influenced Newton (1643)

**Complexity**: O(T²) where T = # temporal objects, filtered by valid type pairs

**Requirements**: Objects must have birth/death metadata

### 6. YonedaCandidates

**Insight**: Hom-pattern overlap

Objects with similar outgoing morphism patterns (Yoneda lemma).

**Example**: Two physicists who both influenced the same 5 others

**Complexity**: O(N²·D) worst case, but fast-filtered by shared targets

## Performance

On physics dataset (57 objects, 69 edges):
- **Candidates surfaced**: 1,232
- **After filtering**: 20 high-confidence
- **Time**: 44 seconds (including embedding inference)
- **Precision**: 80% (8/10 validated)

### Bottlenecks

1. **Embedding inference** (if not cached): 60-70% of time
2. **Oracle.predict() calls**: 1,232 × ~30ms = ~40s
3. **Coherence checking**: O(N²) but N is small after filtering

### Optimization Strategies

- Cache embeddings: 10x speedup
- Batch Oracle predictions: 2-3x speedup
- Parallelize candidate generation: 2x speedup
- Use approximate nearest neighbors (FAISS): 5x speedup for large graphs

## When to Use

### Good Use Cases

- **Scientific literature mining**: "Which papers should cite each other?"
- **Knowledge graph completion**: "Which facts are missing from Wikidata?"
- **Drug discovery**: "Which proteins likely interact?"
- **Theorem proving**: "Which lemmas likely imply this conjecture?"

### Poor Use Cases

- **Dense graphs**: If >80% of possible edges exist, little to conjecture
- **Random graphs**: No structure to exploit
- **Very small graphs**: Need at least ~50 nodes for patterns to emerge

## Extending with New Generators

```python
from oracle.conjecture import _CandidateGenerator

class MyCustomGenerator(_CandidateGenerator):
    name = "my_generator"

    def generate(self) -> Set[Tuple[str, str]]:
        # Access shared cache
        outgoing = self.cache.outgoing
        existing = self.cache.existing

        candidates = set()
        # Your logic here
        for source, mors in outgoing.items():
            # ... find missing pairs ...
            if (source, target) not in existing:
                candidates.add((source, target))

        return candidates

# Use it
engine = ConjectureEngine(oracle)
engine._generators.append(MyCustomGenerator(engine._cache))
result = engine.conjecture()
```

## Testing

```bash
cd KOMPOSOS-III
python -m pytest tests/test_conjecture.py -v
```

All 40 tests should pass.

## Troubleshooting

### "CategoricalOracle requires initialized embeddings"

Embeddings are mandatory. Compute them first:

```python
from data.embeddings import EmbeddingsEngine

embeddings = EmbeddingsEngine()
# Will download model on first run (~400MB)
```

### "No conjectures generated"

Possible causes:
1. **Graph is complete**: All plausible edges already exist
2. **Confidence threshold too high**: Lower `min_confidence`
3. **No embeddings computed**: Some generators won't run

### "OOM during embedding computation"

Use CPU-only mode:

```python
embeddings = EmbeddingsEngine(device='cpu')
```

Or batch process:
```python
for obj in objects:
    embeddings.embed(obj.name)  # Caches automatically
```

## Citation

If you use the Conjecture Engine in research:

```bibtex
@software{komposos3_conjecture,
  title = {KOMPOSOS-III Conjecture Engine},
  author = {[Your Name]},
  year = {2026},
  url = {https://github.com/[your-repo]/KOMPOSOS-III}
}
```

## License

Apache 2.0 — see LICENSE file

---

**Last Updated**: January 30, 2026
**Version**: 0.1.0
**Status**: Research prototype
