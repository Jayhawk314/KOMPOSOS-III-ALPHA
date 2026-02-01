#!/usr/bin/env python3
"""
KOMPOSOS-III Scalability Benchmark
===================================

Measures actual performance characteristics:
1. Query time vs graph size
2. Memory usage vs object count
3. Path enumeration limits
4. Oracle prediction scaling

This provides concrete data for planning scale improvements.
"""

import sys
import time
import tracemalloc
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import create_memory_store, KomposOSStore, StoredObject, StoredMorphism
from evaluation.physics_dataset import create_physics_dataset

# Try to import Oracle
try:
    from oracle import CategoricalOracle
    from data import EmbeddingsEngine, StoreEmbedder
    ORACLE_AVAILABLE = True
except ImportError:
    ORACLE_AVAILABLE = False

# Try to import homotopy
try:
    from hott import check_path_homotopy
    HOMOTOPY_AVAILABLE = True
except ImportError:
    HOMOTOPY_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    graph_size: int  # objects + morphisms
    objects: int
    morphisms: int
    time_ms: float
    memory_mb: float
    result_count: int
    notes: str = ""


def create_scaled_dataset(num_objects: int, avg_morphisms_per_object: int = 3) -> KomposOSStore:
    """
    Create a synthetic dataset of specified size.

    Args:
        num_objects: Number of objects to create
        avg_morphisms_per_object: Average outgoing morphisms per object
    """
    store = create_memory_store()

    # Create objects
    for i in range(num_objects):
        store.add_object(StoredObject(
            name=f"Object_{i}",
            type_name=random.choice(["TypeA", "TypeB", "TypeC"]),
            metadata={"index": i, "birth": 1800 + (i % 200)},
            provenance="synthetic"
        ))

    # Create morphisms (random connections)
    total_morphisms = num_objects * avg_morphisms_per_object
    relation_types = ["influenced", "related_to", "transformed", "extended", "derived"]

    for _ in range(total_morphisms):
        source_idx = random.randint(0, num_objects - 1)
        target_idx = random.randint(0, num_objects - 1)
        if source_idx != target_idx:
            store.add_morphism(StoredMorphism(
                name=random.choice(relation_types),
                source_name=f"Object_{source_idx}",
                target_name=f"Object_{target_idx}",
                confidence=random.uniform(0.5, 1.0),
                provenance="synthetic"
            ))

    return store


def benchmark_path_finding(store: KomposOSStore, source: str, target: str,
                           max_length: int = 5) -> BenchmarkResult:
    """Benchmark path finding between two objects."""
    stats = store.get_statistics()

    tracemalloc.start()
    start_time = time.perf_counter()

    paths = store.find_paths(source, target, max_length=max_length)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        name=f"Path Finding ({source} -> {target}, max={max_length})",
        graph_size=stats["objects"] + stats["morphisms"],
        objects=stats["objects"],
        morphisms=stats["morphisms"],
        time_ms=elapsed_ms,
        memory_mb=peak / 1024 / 1024,
        result_count=len(paths),
        notes=f"Found {len(paths)} paths"
    )


def benchmark_morphism_query(store: KomposOSStore, object_name: str) -> BenchmarkResult:
    """Benchmark morphism queries (outgoing + incoming)."""
    stats = store.get_statistics()

    tracemalloc.start()
    start_time = time.perf_counter()

    outgoing = store.get_morphisms_from(object_name)
    incoming = store.get_morphisms_to(object_name)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        name=f"Morphism Query ({object_name})",
        graph_size=stats["objects"] + stats["morphisms"],
        objects=stats["objects"],
        morphisms=stats["morphisms"],
        time_ms=elapsed_ms,
        memory_mb=peak / 1024 / 1024,
        result_count=len(outgoing) + len(incoming),
        notes=f"Out: {len(outgoing)}, In: {len(incoming)}"
    )


def benchmark_full_load(store: KomposOSStore) -> BenchmarkResult:
    """Benchmark loading all objects and morphisms."""
    stats = store.get_statistics()

    tracemalloc.start()
    start_time = time.perf_counter()

    objects = store.list_objects(limit=100000)
    morphisms = store.list_morphisms(limit=100000)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        name="Full Graph Load",
        graph_size=stats["objects"] + stats["morphisms"],
        objects=stats["objects"],
        morphisms=stats["morphisms"],
        time_ms=elapsed_ms,
        memory_mb=peak / 1024 / 1024,
        result_count=len(objects) + len(morphisms),
        notes=f"Loaded {len(objects)} objects, {len(morphisms)} morphisms"
    )


def benchmark_homotopy(paths: List[List[str]]) -> BenchmarkResult:
    """Benchmark homotopy checking on a set of paths."""
    if not HOMOTOPY_AVAILABLE:
        return BenchmarkResult(
            name="Homotopy Check",
            graph_size=0, objects=0, morphisms=0,
            time_ms=0, memory_mb=0, result_count=0,
            notes="Homotopy module not available"
        )

    tracemalloc.start()
    start_time = time.perf_counter()

    result = check_path_homotopy(paths)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        name=f"Homotopy Check ({len(paths)} paths)",
        graph_size=len(paths),
        objects=len(paths),
        morphisms=0,
        time_ms=elapsed_ms,
        memory_mb=peak / 1024 / 1024,
        result_count=result.num_classes,
        notes=f"{result.num_classes} homotopy classes, all_homotopic={result.all_homotopic}"
    )


def benchmark_oracle(store: KomposOSStore, source: str, target: str) -> BenchmarkResult:
    """Benchmark Oracle prediction."""
    if not ORACLE_AVAILABLE:
        return BenchmarkResult(
            name="Oracle Prediction",
            graph_size=0, objects=0, morphisms=0,
            time_ms=0, memory_mb=0, result_count=0,
            notes="Oracle module not available"
        )

    stats = store.get_statistics()

    try:
        # Initialize embeddings
        embeddings = EmbeddingsEngine()
        if not embeddings.is_available:
            return BenchmarkResult(
                name="Oracle Prediction",
                graph_size=stats["objects"] + stats["morphisms"],
                objects=stats["objects"],
                morphisms=stats["morphisms"],
                time_ms=0, memory_mb=0, result_count=0,
                notes="Embeddings not available (sentence-transformers not installed)"
            )

        # Embed objects (this is part of setup, not the benchmark)
        embedder = StoreEmbedder(store, embeddings)
        embedder.embed_all_objects(show_progress=False)

        # Now benchmark Oracle
        tracemalloc.start()
        start_time = time.perf_counter()

        oracle = CategoricalOracle(store, embeddings)
        result = oracle.predict(source, target)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return BenchmarkResult(
            name=f"Oracle Prediction ({source} -> {target})",
            graph_size=stats["objects"] + stats["morphisms"],
            objects=stats["objects"],
            morphisms=stats["morphisms"],
            time_ms=elapsed_ms,
            memory_mb=peak / 1024 / 1024,
            result_count=len(result.predictions),
            notes=f"{len(result.predictions)} predictions, {result.total_candidates} candidates"
        )
    except Exception as e:
        return BenchmarkResult(
            name="Oracle Prediction",
            graph_size=stats["objects"] + stats["morphisms"],
            objects=stats["objects"],
            morphisms=stats["morphisms"],
            time_ms=0, memory_mb=0, result_count=0,
            notes=f"Error: {e}"
        )


def run_scaling_tests() -> List[BenchmarkResult]:
    """Run benchmarks at different scales."""
    results = []

    print("=" * 70)
    print("KOMPOSOS-III SCALABILITY BENCHMARK")
    print("=" * 70)
    print()

    # Test 1: Physics dataset (baseline)
    print("[1] Physics Dataset (Baseline)")
    print("-" * 40)
    physics_store = create_physics_dataset()
    stats = physics_store.get_statistics()
    print(f"    Objects: {stats['objects']}")
    print(f"    Morphisms: {stats['morphisms']}")
    print()

    # Path finding tests
    test_pairs = [
        ("Planck", "Feynman"),
        ("Newton", "Dirac"),
        ("Galileo", "QuantumMechanics"),
    ]

    for source, target in test_pairs:
        for max_len in [5, 8, 10]:
            result = benchmark_path_finding(physics_store, source, target, max_len)
            results.append(result)
            print(f"    {result.name}")
            print(f"      Time: {result.time_ms:.2f} ms")
            print(f"      Memory: {result.memory_mb:.2f} MB")
            print(f"      Paths found: {result.result_count}")
            print()

    # Morphism query test
    for obj in ["Einstein", "Dirac", "Newton"]:
        result = benchmark_morphism_query(physics_store, obj)
        results.append(result)
        print(f"    {result.name}")
        print(f"      Time: {result.time_ms:.2f} ms")
        print(f"      {result.notes}")
        print()

    # Full load test
    result = benchmark_full_load(physics_store)
    results.append(result)
    print(f"    {result.name}")
    print(f"      Time: {result.time_ms:.2f} ms")
    print(f"      Memory: {result.memory_mb:.2f} MB")
    print()

    # Homotopy test
    paths = physics_store.find_paths("Planck", "Feynman", max_length=8)
    path_lists = []
    for p in paths[:50]:  # Limit to 50 for benchmark
        # Convert morphism IDs to node names
        nodes = [p.source_name]
        for mor_id in p.morphism_ids:
            mor = physics_store.get_morphism(mor_id)
            if mor:
                nodes.append(mor.target_name)
        path_lists.append(nodes)

    if path_lists:
        result = benchmark_homotopy(path_lists)
        results.append(result)
        print(f"    {result.name}")
        print(f"      Time: {result.time_ms:.2f} ms")
        print(f"      {result.notes}")
        print()

    # Oracle test
    print("[2] Oracle Prediction Benchmark")
    print("-" * 40)
    result = benchmark_oracle(physics_store, "Planck", "Feynman")
    results.append(result)
    print(f"    {result.name}")
    print(f"      Time: {result.time_ms:.2f} ms")
    print(f"      Memory: {result.memory_mb:.2f} MB")
    print(f"      {result.notes}")
    print()

    # Test 2: Scaled synthetic datasets
    print("[3] Scaling Tests (Synthetic Data)")
    print("-" * 40)

    scale_sizes = [100, 500, 1000, 2000, 5000]

    for size in scale_sizes:
        print(f"\n  Dataset Size: {size} objects")

        scaled_store = create_scaled_dataset(size, avg_morphisms_per_object=3)
        stats = scaled_store.get_statistics()
        print(f"    Actual: {stats['objects']} objects, {stats['morphisms']} morphisms")

        # Path finding (random pair)
        source = f"Object_0"
        target = f"Object_{size-1}"

        result = benchmark_path_finding(scaled_store, source, target, max_length=5)
        results.append(result)
        print(f"    Path Finding (max_length=5):")
        print(f"      Time: {result.time_ms:.2f} ms")
        print(f"      Paths: {result.result_count}")

        # Full load
        result = benchmark_full_load(scaled_store)
        results.append(result)
        print(f"    Full Load:")
        print(f"      Time: {result.time_ms:.2f} ms")
        print(f"      Memory: {result.memory_mb:.2f} MB")

    return results


def analyze_results(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Analyze benchmark results and identify bottlenecks."""
    analysis = {
        "total_benchmarks": len(results),
        "slowest_operations": [],
        "memory_intensive": [],
        "scaling_observations": [],
    }

    # Find slowest operations
    sorted_by_time = sorted(results, key=lambda r: r.time_ms, reverse=True)
    analysis["slowest_operations"] = [
        {"name": r.name, "time_ms": r.time_ms, "size": r.graph_size}
        for r in sorted_by_time[:5]
    ]

    # Find memory-intensive operations
    sorted_by_memory = sorted(results, key=lambda r: r.memory_mb, reverse=True)
    analysis["memory_intensive"] = [
        {"name": r.name, "memory_mb": r.memory_mb, "size": r.graph_size}
        for r in sorted_by_memory[:5]
    ]

    # Analyze scaling (path finding at different sizes)
    path_results = [r for r in results if "Path Finding" in r.name and "Object_0" in r.name]
    if len(path_results) >= 2:
        sizes = [r.objects for r in path_results]
        times = [r.time_ms for r in path_results]

        if len(sizes) >= 2 and sizes[-1] > sizes[0]:
            size_ratio = sizes[-1] / sizes[0]
            time_ratio = times[-1] / max(times[0], 0.01)

            if time_ratio > size_ratio * 2:
                analysis["scaling_observations"].append(
                    f"Path finding scales SUPER-LINEARLY: {size_ratio:.1f}x size -> {time_ratio:.1f}x time"
                )
            elif time_ratio > size_ratio:
                analysis["scaling_observations"].append(
                    f"Path finding scales worse than linear: {size_ratio:.1f}x size -> {time_ratio:.1f}x time"
                )
            else:
                analysis["scaling_observations"].append(
                    f"Path finding scales sub-linearly: {size_ratio:.1f}x size -> {time_ratio:.1f}x time"
                )

    return analysis


def generate_report(results: List[BenchmarkResult], analysis: Dict) -> str:
    """Generate markdown benchmark report."""
    report = """# KOMPOSOS-III Scalability Benchmark Report

## Executive Summary

This report measures actual performance characteristics to guide scaling decisions.

---

## Benchmark Results

### Performance by Operation

| Operation | Graph Size | Time (ms) | Memory (MB) | Results |
|-----------|------------|-----------|-------------|---------|
"""

    for r in results:
        report += f"| {r.name[:50]} | {r.graph_size} | {r.time_ms:.2f} | {r.memory_mb:.2f} | {r.result_count} |\n"

    report += """
---

## Analysis

### Slowest Operations

"""
    for op in analysis.get("slowest_operations", []):
        report += f"- **{op['name']}**: {op['time_ms']:.2f} ms (size: {op['size']})\n"

    report += """
### Memory-Intensive Operations

"""
    for op in analysis.get("memory_intensive", []):
        report += f"- **{op['name']}**: {op['memory_mb']:.2f} MB (size: {op['size']})\n"

    report += """
### Scaling Observations

"""
    for obs in analysis.get("scaling_observations", []):
        report += f"- {obs}\n"

    report += """
---

## Bottleneck Analysis

### Current Limits (Estimated)

| Component | Comfortable | Maximum | Bottleneck |
|-----------|-------------|---------|------------|
| Objects | ~10,000 | ~100,000 | Memory |
| Morphisms | ~50,000 | ~500,000 | Index queries |
| Path Finding | ~1,000 paths | ~10,000 paths | Exponential BFS |
| Homotopy | ~100 paths | ~1,000 paths | O(n^2) pairwise |

### Recommendations for Scale

1. **Short-term**: Add path caching, early termination, bidirectional BFS
2. **Medium-term**: Lazy path iteration, incremental Kan extensions
3. **Long-term**: Graph database (Neo4j), approximate algorithms

---

## Path to True Scale

### Phase 1: Optimizations (Current Architecture)
- Path memoization
- Query result caching
- Index optimization

### Phase 2: Approximate Methods
- Sampling-based path enumeration
- Locality-sensitive hashing for similarity
- Probabilistic homotopy checking

### Phase 3: Infrastructure Change
- Graph database backend (Neo4j/ArangoDB)
- Distributed computation
- Native graph traversal

---

*Benchmark generated by KOMPOSOS-III Scalability Test*
"""

    return report


if __name__ == "__main__":
    results = run_scaling_tests()

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    analysis = analyze_results(results)

    print("\nSlowest Operations:")
    for op in analysis["slowest_operations"]:
        print(f"  - {op['name']}: {op['time_ms']:.2f} ms")

    print("\nMemory-Intensive Operations:")
    for op in analysis["memory_intensive"]:
        print(f"  - {op['name']}: {op['memory_mb']:.2f} MB")

    print("\nScaling Observations:")
    for obs in analysis["scaling_observations"]:
        print(f"  - {obs}")

    # Generate report
    report = generate_report(results, analysis)
    output_path = Path(__file__).parent.parent / "evaluation_reports" / "SCALABILITY_BENCHMARK.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding='utf-8')

    print(f"\nDetailed report saved to: {output_path}")
