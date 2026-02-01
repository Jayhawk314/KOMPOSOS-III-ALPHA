#!/usr/bin/env python3
"""
KOMPOSOS-III Comprehensive Demonstration
=========================================

Demonstrates all system capabilities on the physics evolution dataset.
Run with: python demo.py
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from evaluation.physics_dataset import create_physics_dataset
from geometry import OllivierRicciCurvature, DiscreteRicciFlow
from hott import check_path_homotopy, check_geometric_homotopy
from oracle.strategies import (
    KanExtensionStrategy, TemporalReasoningStrategy, TypeHeuristicStrategy,
    YonedaPatternStrategy, CompositionStrategy, StructuralHoleStrategy,
    GeometricStrategy
)


def run_demo():
    output = []

    def log(msg=""):
        print(msg)
        output.append(msg)

    log("=" * 70)
    log("KOMPOSOS-III COMPREHENSIVE DEMONSTRATION")
    log("=" * 70)
    log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log()

    # Create dataset
    store = create_physics_dataset()
    stats = store.get_statistics()
    log(f"DATASET: {stats['objects']} objects, {stats['morphisms']} morphisms")
    log()

    # ========================================================================
    # 1. OLLIVIER-RICCI CURVATURE
    # ========================================================================
    log("1. OLLIVIER-RICCI CURVATURE ANALYSIS")
    log("-" * 50)
    ricci = OllivierRicciCurvature(store)
    curv = ricci.compute_all_curvatures()

    log(f"   Edges analyzed: {len(curv.edge_curvatures)}")
    log(f"   Mean curvature: {curv.statistics['mean']:.4f}")
    log(f"   Spherical (clusters): {curv.num_spherical}")
    log(f"   Hyperbolic (bridges): {curv.num_hyperbolic}")
    log(f"   Euclidean (chains): {curv.num_euclidean}")

    sorted_edges = sorted(curv.edge_curvatures.items(), key=lambda x: x[1])
    log()
    log("   TOP CLUSTERS (positive curvature):")
    for (s, t), k in sorted_edges[-3:]:
        log(f"     {s} <-> {t}: k={k:.3f}")
    log()
    log("   TOP BRIDGES (negative curvature):")
    for (s, t), k in sorted_edges[:3]:
        log(f"     {s} <-> {t}: k={k:.3f}")
    log()

    # ========================================================================
    # 2. DISCRETE RICCI FLOW
    # ========================================================================
    log("2. DISCRETE RICCI FLOW DECOMPOSITION")
    log("-" * 50)
    flow = DiscreteRicciFlow(store)
    decomp = flow.flow(max_steps=25, dt=0.15)

    log(f"   Steps: {decomp.num_steps}")
    log(f"   Converged: {decomp.converged}")
    log(f"   Regions: {decomp.num_regions}")
    log(f"   Boundary edges: {len(decomp.boundary_edges)}")
    log()
    log("   GEOMETRIC REGIONS:")
    for region in decomp.regions[:6]:
        nodes = list(region.nodes)[:4]
        log(f"     {region.name} ({region.geometry_type.value}): {region.size} nodes")
        log(f"       e.g. {nodes}")
    log()

    # ========================================================================
    # 3. PATH FINDING
    # ========================================================================
    log("3. PATH FINDING")
    log("-" * 50)

    queries = [
        ("Newton", "Schrodinger"),
        ("Planck", "Feynman"),
        ("Maxwell", "QED"),
    ]

    all_paths = {}
    for src, tgt in queries:
        paths = store.find_paths(src, tgt, max_length=6)
        all_paths[(src, tgt)] = paths
        log(f"   {src} -> {tgt}: {len(paths)} paths")
    log()

    # ========================================================================
    # 4. STANDARD HOMOTOPY ANALYSIS
    # ========================================================================
    log("4. STANDARD HOMOTOPY ANALYSIS")
    log("-" * 50)

    src, tgt = "Newton", "Schrodinger"
    paths = all_paths[(src, tgt)]
    path_seqs = []
    if paths:
        for p in paths[:8]:
            seq = [src]
            for mid in p.morphism_ids:
                m = store.get_morphism(mid)
                if m and m.target_name not in seq:
                    seq.append(m.target_name)
            path_seqs.append(seq)

        hom_result = check_path_homotopy(path_seqs, store)
        log(f"   Paths analyzed: {len(path_seqs)}")
        log(f"   Homotopy classes: {hom_result.num_classes}")
        log(f"   All homotopic: {hom_result.all_homotopic}")
        if hom_result.shared_spine:
            log(f"   Shared spine: {hom_result.shared_spine}")
    log()

    # ========================================================================
    # 5. GEOMETRIC HOMOTOPY ANALYSIS
    # ========================================================================
    log("5. GEOMETRIC HOMOTOPY ANALYSIS (Thurston-aware)")
    log("-" * 50)

    if path_seqs:
        geo_result = check_geometric_homotopy(path_seqs, ricci=ricci)
        log(f"   Geometric classes: {geo_result.num_classes}")
        log(f"   All geometrically homotopic: {geo_result.all_homotopic}")
        log()
        log("   PATH SIGNATURES:")
        for i, sig in enumerate(geo_result.signatures[:4]):
            sig_str = " -> ".join(sig.simplified)
            log(f"     Path {i+1}: {sig_str}")
    log()

    # ========================================================================
    # 6. ORACLE STRATEGIES
    # ========================================================================
    log("6. ORACLE PREDICTION STRATEGIES")
    log("-" * 50)

    strategies = [
        ("Kan Extension", KanExtensionStrategy(store)),
        ("Temporal Reasoning", TemporalReasoningStrategy(store)),
        ("Type Heuristics", TypeHeuristicStrategy(store)),
        ("Yoneda Pattern", YonedaPatternStrategy(store)),
        ("Composition", CompositionStrategy(store)),
        ("Structural Hole", StructuralHoleStrategy(store)),
        ("Geometric", GeometricStrategy(store, curvature_computer=ricci)),
    ]

    test_pairs = [
        ("Newton", "Lagrange"),
        ("Einstein", "Hilbert"),
        ("Bohr", "Heisenberg"),
        ("Feynman", "StringTheory"),
    ]

    log("   Testing predictions for multiple concept pairs...")
    log()

    total_preds = 0
    all_predictions = []
    for name, strategy in strategies:
        strat_preds = 0
        for src, tgt in test_pairs:
            preds = strategy.predict(src, tgt)
            strat_preds += len(preds)
            for p in preds:
                all_predictions.append((name, p))
        total_preds += strat_preds
        status = "active" if strat_preds > 0 else "no predictions"
        log(f"   {name}: {strat_preds} predictions ({status})")

    log()
    log(f"   TOTAL PREDICTIONS: {total_preds}")
    log()

    # Show sample predictions
    log("   SAMPLE PREDICTIONS:")
    shown = 0
    for name, p in all_predictions[:5]:
        reason = p.reasoning[:55].replace("\n", " ")
        log(f"     [{name}] {p.source}->{p.target}: {p.confidence:.2f}")
        log(f"       {reason}...")
        shown += 1
    log()

    # ========================================================================
    # 7. EQUIVALENCE CLASSES
    # ========================================================================
    log("7. EQUIVALENCE CLASSES (HoTT)")
    log("-" * 50)

    try:
        equivs = store.list_equivalences()
        log(f"   Equivalence classes: {len(equivs)}")
        for eq in equivs[:4]:
            members = ", ".join(eq.member_names[:3])
            log(f"     {eq.equivalence_type}: {members}")
    except Exception as e:
        log(f"   (Equivalence listing: {e})")
    log()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    log("=" * 70)
    log("DEMONSTRATION COMPLETE")
    log("=" * 70)
    log()
    log("CAPABILITIES DEMONSTRATED:")
    log("  [x] Data loading (physics evolution dataset)")
    log("  [x] Ollivier-Ricci curvature computation")
    log("  [x] Discrete Ricci flow decomposition")
    log("  [x] Path finding between concepts")
    log("  [x] Standard path homotopy analysis")
    log("  [x] Geometric (Thurston-aware) homotopy")
    log("  [x] 7 Oracle prediction strategies")
    log("  [x] HoTT equivalence class tracking")
    log()
    log("MATHEMATICAL FOUNDATIONS:")
    log("  - Category Theory (morphisms, composition, Kan extensions)")
    log("  - Homotopy Type Theory (paths, identity types, homotopy)")
    log("  - Differential Geometry (Ricci curvature, Ricci flow)")
    log("  - Thurston Geometrization (spherical/hyperbolic/euclidean)")
    log()

    return "\n".join(output)


if __name__ == "__main__":
    result = run_demo()

    # Save to markdown
    with open("DEMO_OUTPUT.md", "w", encoding="utf-8") as f:
        f.write("# KOMPOSOS-III Demonstration Output\n\n")
        f.write("```\n")
        f.write(result)
        f.write("\n```\n")

    print()
    print(f"Output saved to: DEMO_OUTPUT.md")
