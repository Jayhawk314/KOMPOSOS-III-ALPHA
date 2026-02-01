# KOMPOSOS-III Scaling Roadmap

## Current Performance (Benchmark Results)

Based on actual measurements from `evaluation/scalability_benchmark.py`:

| Scale | Objects | Morphisms | Full Load | Path Finding | Memory |
|-------|---------|-----------|-----------|--------------|--------|
| Physics | 57 | 69 | 2.5 ms | 3-10 ms | 0.15 MB |
| Small | 100 | 295 | 7 ms | 7 ms | 0.31 MB |
| Medium | 1K | 3K | 70 ms | 7 ms | 3.3 MB |
| Large | 5K | 15K | 381 ms | 6 ms | 17 MB |

**Key Finding**: Path finding is NOT the bottleneck - it's full graph load.

---

## Phase 1: Quick Wins (1-2 days)

### 1.1 Lazy Loading

**Problem**: Full graph load is O(n) and loads everything into memory.

**Solution**: Load on-demand with caching.

```python
class LazyStore:
    """Wrapper that loads data only when needed."""

    def __init__(self, store):
        self.store = store
        self._object_cache = {}
        self._morphism_cache = {}

    def get_object(self, name):
        if name not in self._object_cache:
            self._object_cache[name] = self.store.get_object(name)
        return self._object_cache[name]

    def get_morphisms_from(self, source):
        if source not in self._morphism_cache:
            self._morphism_cache[source] = self.store.get_morphisms_from(source)
        return self._morphism_cache[source]
```

**Expected Improvement**: 10-100x for targeted queries.

### 1.2 Path Result Limit

**Problem**: Path finding can return thousands of paths.

**Solution**: Add early termination.

```python
def find_paths(source, target, max_length=5, max_paths=100):
    """Stop after finding max_paths."""
    paths = []
    for path in path_generator(source, target, max_length):
        paths.append(path)
        if len(paths) >= max_paths:
            break
    return paths
```

**Expected Improvement**: Bounded worst-case time.

### 1.3 Index Optimization

**Problem**: SQLite indexes exist but aren't always used.

**Solution**: Add composite indexes.

```sql
CREATE INDEX idx_morphisms_source_target ON morphisms(source_name, target_name);
CREATE INDEX idx_morphisms_confidence ON morphisms(confidence DESC);
```

**Expected Improvement**: 2-5x for filtered queries.

---

## Phase 2: Algorithmic Improvements (1 week)

### 2.1 Bidirectional BFS

**Problem**: BFS explores all nodes at each depth level.

**Solution**: Search from both ends, meet in middle.

```python
def bidirectional_path_find(source, target, max_length):
    """Meet-in-the-middle BFS."""
    forward = {source: []}
    backward = {target: []}

    for depth in range(max_length // 2):
        # Expand forward frontier
        expand_frontier(forward, outgoing_morphisms)
        # Expand backward frontier
        expand_frontier(backward, incoming_morphisms)
        # Check for intersection
        common = set(forward.keys()) & set(backward.keys())
        if common:
            return construct_paths(forward, backward, common)
```

**Expected Improvement**: O(b^(d/2)) instead of O(b^d).

### 2.2 Incremental Kan Extensions

**Problem**: Kan extension recomputes comma category each time.

**Solution**: Cache and update incrementally.

```python
class IncrementalKanExtension:
    def __init__(self):
        self._comma_cache = {}

    def extend(self, target):
        if target.name in self._comma_cache:
            return self._comma_cache[target.name]

        result = self._compute_extension(target)
        self._comma_cache[target.name] = result
        return result

    def invalidate(self, changed_morphism):
        """Invalidate affected cache entries."""
        affected = self._find_affected(changed_morphism)
        for key in affected:
            del self._comma_cache[key]
```

### 2.3 Sampling-Based Homotopy

**Problem**: O(n^2) pairwise path comparison.

**Solution**: Sample representative paths.

```python
def sample_homotopy_check(paths, sample_size=50):
    """Check homotopy on a sample, then verify."""
    if len(paths) <= sample_size:
        return full_homotopy_check(paths)

    # Sample diverse paths (different lengths, intermediates)
    sample = diverse_sample(paths, sample_size)

    # Check sample
    sample_result = full_homotopy_check(sample)

    # Extrapolate with confidence interval
    return HomotopyResult(
        approximate=True,
        confidence=0.95,
        estimated_classes=sample_result.num_classes,
        sample_size=sample_size
    )
```

---

## Phase 3: Infrastructure (2-4 weeks)

### 3.1 Graph Database Backend

**Why Neo4j?**
- Native graph traversal (orders of magnitude faster for path queries)
- Cypher query language designed for relationships
- Built-in indexing and caching
- Scales to billions of nodes

**Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                    KOMPOSOS-III                              │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │  Store Interface │  │         Neo4j Backend            │ │
│  │  (unchanged API) │──│  - Native path traversal         │ │
│  │                  │  │  - Relationship patterns         │ │
│  │                  │  │  - Graph algorithms (PageRank)   │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
│                                                              │
│  Migration Path:                                             │
│  1. Create Neo4jStore implementing KomposOSStore interface   │
│  2. Add graph-native path queries                            │
│  3. Leverage Neo4j's GDS (Graph Data Science) library        │
└─────────────────────────────────────────────────────────────┘
```

**Sample Cypher Queries**:

```cypher
// Find all paths (native, fast)
MATCH path = (a:Object {name: 'Planck'})-[:MORPHISM*1..8]->(b:Object {name: 'Feynman'})
RETURN path

// Find hubs (PageRank)
CALL gds.pageRank.stream('knowledge-graph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC LIMIT 10

// Similarity (Jaccard on neighborhoods)
MATCH (a:Object {name: 'Einstein'})-[:MORPHISM]->(neighbor)
WITH a, collect(neighbor) AS neighborsA
MATCH (b:Object)-[:MORPHISM]->(neighbor)
WHERE b <> a
WITH a, b, neighborsA, collect(neighbor) AS neighborsB
RETURN b.name,
       gds.similarity.jaccard(neighborsA, neighborsB) AS similarity
ORDER BY similarity DESC
```

### 3.2 Approximate Similarity (LSH)

**Problem**: O(n^2) pairwise embedding comparisons.

**Solution**: Locality-Sensitive Hashing.

```python
from datasketch import MinHashLSH

class ApproximateSimilarity:
    def __init__(self, embeddings, threshold=0.7):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=128)
        self._index_embeddings(embeddings)

    def find_similar(self, object_name, k=10):
        """Find k most similar objects in O(1) average."""
        minhash = self._get_minhash(object_name)
        candidates = self.lsh.query(minhash)
        return candidates[:k]
```

**Expected Improvement**: O(1) average instead of O(n).

### 3.3 Distributed Computation

For truly massive graphs (10M+ nodes):

```
┌─────────────────────────────────────────────────────────────┐
│                  Distributed KOMPOSOS-III                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  │ Worker N │    │
│  │ Paths A-G│  │ Paths H-N│  │ Paths O-T│  │ Paths U-Z│    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │             │             │             │           │
│       └─────────────┴──────┬──────┴─────────────┘           │
│                            │                                │
│                     ┌──────▼──────┐                         │
│                     │ Coordinator │                         │
│                     │ Merge paths │                         │
│                     │ Homotopy    │                         │
│                     └─────────────┘                         │
│                                                              │
│  Technologies:                                               │
│  - Apache Spark GraphX                                       │
│  - Dask for distributed Python                               │
│  - Ray for parallel processing                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Recommended Priority

| Priority | Task | Effort | Impact | When |
|----------|------|--------|--------|------|
| **P0** | Lazy loading + caching | 1 day | High | Now |
| **P0** | Path result limit | 2 hours | High | Now |
| **P1** | Bidirectional BFS | 2 days | Medium | Week 1 |
| **P1** | Index optimization | 1 day | Medium | Week 1 |
| **P2** | Incremental Kan | 3 days | Medium | Week 2 |
| **P2** | Sampling homotopy | 2 days | Medium | Week 2 |
| **P3** | Neo4j backend | 1-2 weeks | Very High | Month 1 |
| **P3** | LSH similarity | 3 days | High | Month 1 |
| **P4** | Distributed | 2-4 weeks | Very High | Quarter 2 |

---

## Success Metrics

After implementing Phase 1-2:

| Metric | Current | Target |
|--------|---------|--------|
| Path query (10K graph) | ~10 ms | <5 ms |
| Full load (10K graph) | ~350 ms | <50 ms (lazy) |
| Memory (10K graph) | ~20 MB | <5 MB (lazy) |
| Homotopy (100 paths) | ~10 ms | ~10 ms |

After Phase 3 (Neo4j):

| Metric | Current | Target |
|--------|---------|--------|
| Path query (1M graph) | Not feasible | <100 ms |
| Similarity search | O(n^2) | O(1) with LSH |
| Hub identification | Manual | Native PageRank |

---

## Next Steps

1. **Implement P0 tasks** (lazy loading, path limits) - immediate
2. **Run benchmark again** to measure improvement
3. **Evaluate Neo4j** - set up test instance, migrate physics dataset
4. **Design abstraction layer** - ensure Oracle strategies work with both backends

---

*Scaling Roadmap for KOMPOSOS-III*
*"From 126 items to 1 million"*
