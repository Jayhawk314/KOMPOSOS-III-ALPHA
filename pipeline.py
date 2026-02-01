#!/usr/bin/env python3
"""
KOMPOSOS-III Full Analysis Pipeline
====================================

Run all analysis features on your data in one command:
- Load data
- Compute embeddings
- Compute Ricci curvature
- Run Ricci flow decomposition
- Run Oracle predictions
- Analyze path homotopy (standard + geometric)
- Generate comprehensive report

Usage:
    python pipeline.py --corpus ./my_corpus
    python pipeline.py --corpus ./my_corpus --source Newton --target Einstein
    python pipeline.py --db ./store.db --source A --target B
    python pipeline.py --use-physics  # Use built-in physics dataset for testing
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))


def run_pipeline(
    corpus_path=None,
    db_path=None,
    source=None,
    target=None,
    use_physics=False,
    output_dir=None,
    max_paths=10,
    skip_embeddings=False
):
    """
    Run the full KOMPOSOS-III analysis pipeline.

    Args:
        corpus_path: Path to corpus directory with CSV/JSON files
        db_path: Path to existing database file
        source: Source concept for path analysis
        target: Target concept for path analysis
        use_physics: Use built-in physics dataset
        output_dir: Directory for output reports
        max_paths: Maximum paths to analyze
        skip_embeddings: Skip embedding computation (faster)
    """

    print("=" * 70)
    print("KOMPOSOS-III Full Analysis Pipeline")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # =========================================================================
    # STEP 1: Load/Create Store
    # =========================================================================
    print("[1/8] Loading Data...")
    print("-" * 40)

    if use_physics:
        from evaluation.physics_dataset import create_physics_dataset
        store = create_physics_dataset()
        print("  Using built-in physics dataset")
    elif corpus_path:
        from data import create_store, CorpusLoader
        corpus_path = Path(corpus_path)
        db_file = corpus_path / "store.db"
        store = create_store(db_file)

        loader = CorpusLoader(corpus_path)
        stats = loader.load_all(store)
        print(f"  Loaded from corpus: {corpus_path}")
        print(f"  Objects: {stats.get('objects', 0)}, Morphisms: {stats.get('morphisms', 0)}")
    elif db_path:
        from data import create_store
        store = create_store(Path(db_path))
        print(f"  Loaded existing database: {db_path}")
    else:
        print("  ERROR: Must specify --corpus, --db, or --use-physics")
        return None

    stats = store.get_statistics()
    print(f"  Total: {stats['objects']} objects, {stats['morphisms']} morphisms")
    print()

    # =========================================================================
    # STEP 2: Compute Embeddings (Optional)
    # =========================================================================
    print("[2/8] Computing Embeddings...")
    print("-" * 40)

    if skip_embeddings:
        print("  Skipped (--skip-embeddings flag)")
        embeddings = None
    else:
        try:
            from data import EmbeddingsEngine, StoreEmbedder
            embeddings = EmbeddingsEngine()
            embedder = StoreEmbedder(store, embeddings)
            embedded_count = embedder.embed_all_objects()
            print(f"  Embedded {embedded_count} objects")
        except Exception as e:
            print(f"  Skipped (error: {e})")
            embeddings = None
    print()

    # =========================================================================
    # STEP 3: Compute Ollivier-Ricci Curvature
    # =========================================================================
    print("[3/8] Computing Ollivier-Ricci Curvature...")
    print("-" * 40)

    try:
        from geometry import OllivierRicciCurvature
        ricci = OllivierRicciCurvature(store)
        curvature_result = ricci.compute_all_curvatures()

        print(f"  Edge curvatures: {len(curvature_result.edge_curvatures)}")
        print(f"  Mean curvature: {curvature_result.statistics['mean']:.4f}")
        print(f"  Range: [{curvature_result.statistics['min']:.3f}, {curvature_result.statistics['max']:.3f}]")

        # Show top bridges and clusters
        sorted_edges = sorted(curvature_result.edge_curvatures.items(), key=lambda x: x[1])
        print(f"  Top bridge (hyperbolic): {sorted_edges[0][0]} (k={sorted_edges[0][1]:.3f})")
        print(f"  Top cluster (spherical): {sorted_edges[-1][0]} (k={sorted_edges[-1][1]:.3f})")
    except Exception as e:
        print(f"  Error: {e}")
        ricci = None
        curvature_result = None
    print()

    # =========================================================================
    # STEP 4: Run Discrete Ricci Flow
    # =========================================================================
    print("[4/8] Running Discrete Ricci Flow...")
    print("-" * 40)

    try:
        from geometry import DiscreteRicciFlow
        flow = DiscreteRicciFlow(store)
        decomposition = flow.flow(max_steps=30, dt=0.15)

        print(f"  Converged: {decomposition.converged}")
        print(f"  Steps: {decomposition.num_steps}")
        print(f"  Regions found: {decomposition.num_regions}")
        print(f"  Boundary edges: {len(decomposition.boundary_edges)}")

        # Show region summary
        for region in decomposition.regions[:5]:
            sample = list(region.nodes)[:3]
            print(f"    {region.name}: {region.size} nodes ({region.geometry_type.value})")
    except Exception as e:
        print(f"  Error: {e}")
        decomposition = None
    print()

    # =========================================================================
    # STEP 5: Find Source/Target (if not specified)
    # =========================================================================
    if not source or not target:
        print("[5/8] Selecting Source/Target for Path Analysis...")
        print("-" * 40)

        # Find nodes with most connections
        morphisms = store.list_morphisms(limit=10000)
        out_degree = {}
        in_degree = {}
        for m in morphisms:
            out_degree[m.source_name] = out_degree.get(m.source_name, 0) + 1
            in_degree[m.target_name] = in_degree.get(m.target_name, 0) + 1

        # Source = high out-degree, Target = high in-degree
        if out_degree and in_degree:
            source = max(out_degree, key=out_degree.get) if not source else source
            target = max(in_degree, key=in_degree.get) if not target else target
            print(f"  Auto-selected: {source} -> {target}")
        else:
            print("  No morphisms found, skipping path analysis")
            source, target = None, None
    else:
        print(f"[5/8] Using specified: {source} -> {target}")
    print()

    # =========================================================================
    # STEP 6: Path Finding & Homotopy Analysis
    # =========================================================================
    print("[6/8] Path Finding & Homotopy Analysis...")
    print("-" * 40)

    paths = []
    path_sequences = []
    homotopy_result = None
    geo_homotopy_result = None

    if source and target:
        paths = store.find_paths(source, target, max_length=8)
        print(f"  Paths found: {len(paths)}")

        if paths:
            # Extract node sequences
            for path in paths[:max_paths]:
                sequence = [source]
                for mor_id in path.morphism_ids:
                    mor = store.get_morphism(mor_id)
                    if mor and mor.target_name not in sequence:
                        sequence.append(mor.target_name)
                path_sequences.append(sequence)

            # Standard homotopy
            try:
                from hott import check_path_homotopy
                homotopy_result = check_path_homotopy(path_sequences, store)
                print(f"  Standard homotopy classes: {homotopy_result.num_classes}")
                print(f"  All homotopic: {homotopy_result.all_homotopic}")
            except Exception as e:
                print(f"  Standard homotopy error: {e}")

            # Geometric homotopy
            try:
                from hott import check_geometric_homotopy
                geo_homotopy_result = check_geometric_homotopy(path_sequences, ricci=ricci)
                print(f"  Geometric homotopy classes: {geo_homotopy_result.num_classes}")
                print(f"  All geometrically homotopic: {geo_homotopy_result.all_homotopic}")
            except Exception as e:
                print(f"  Geometric homotopy error: {e}")
        else:
            print(f"  No paths from {source} to {target}")
    print()

    # =========================================================================
    # STEP 7: Oracle Predictions
    # =========================================================================
    print("[7/8] Running Oracle Predictions...")
    print("-" * 40)

    predictions = []
    if source and target:
        try:
            from oracle import CategoricalOracle
            oracle = CategoricalOracle(store, embeddings=embeddings)
            oracle_result = oracle.predict(source, target)
            predictions = oracle_result.predictions

            print(f"  Predictions generated: {len(predictions)}")
            for pred in predictions[:5]:
                print(f"    {pred.source} -> {pred.target}: {pred.confidence:.2f} ({pred.strategy_name})")
        except Exception as e:
            print(f"  Oracle error: {e}")

            # Fallback: try individual strategies
            try:
                from oracle.strategies import create_all_strategies
                strategies = create_all_strategies(store, embeddings)
                print(f"  Using {len(strategies)} individual strategies...")

                for strategy in strategies[:5]:
                    try:
                        preds = strategy.predict(source, target)
                        predictions.extend(preds)
                    except:
                        pass

                print(f"  Predictions from strategies: {len(predictions)}")
            except Exception as e2:
                print(f"  Strategy fallback error: {e2}")
    print()

    # =========================================================================
    # STEP 8: Generate Report
    # =========================================================================
    print("[8/8] Generating Report...")
    print("-" * 40)

    report_lines = []
    report_lines.append(f"# KOMPOSOS-III Analysis Report")
    report_lines.append(f"")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"")

    # Dataset summary
    report_lines.append(f"## Dataset Summary")
    report_lines.append(f"- Objects: {stats['objects']}")
    report_lines.append(f"- Morphisms: {stats['morphisms']}")
    report_lines.append(f"")

    # Geometry summary
    if curvature_result:
        report_lines.append(f"## Geometric Analysis")
        report_lines.append(f"- Edge curvatures computed: {len(curvature_result.edge_curvatures)}")
        report_lines.append(f"- Mean curvature: {curvature_result.statistics['mean']:.4f}")
        report_lines.append(f"- Spherical edges (clusters): {curvature_result.num_spherical}")
        report_lines.append(f"- Hyperbolic edges (bridges): {curvature_result.num_hyperbolic}")
        report_lines.append(f"- Euclidean edges (chains): {curvature_result.num_euclidean}")
        report_lines.append(f"")

    if decomposition:
        report_lines.append(f"## Ricci Flow Decomposition")
        report_lines.append(f"- Regions: {decomposition.num_regions}")
        report_lines.append(f"- Boundary edges: {len(decomposition.boundary_edges)}")
        report_lines.append(f"- Converged: {decomposition.converged}")
        report_lines.append(f"")

        report_lines.append(f"### Regions")
        for region in decomposition.regions[:10]:
            sample = ", ".join(list(region.nodes)[:5])
            report_lines.append(f"- **{region.name}** ({region.geometry_type.value}): {region.size} nodes")
            report_lines.append(f"  - Sample: {sample}")
        report_lines.append(f"")

    # Path analysis
    if source and target:
        report_lines.append(f"## Path Analysis: {source} -> {target}")
        report_lines.append(f"- Paths found: {len(paths)}")

        if homotopy_result:
            report_lines.append(f"- Standard homotopy classes: {homotopy_result.num_classes}")
        if geo_homotopy_result:
            report_lines.append(f"- Geometric homotopy classes: {geo_homotopy_result.num_classes}")
        report_lines.append(f"")

        if path_sequences:
            report_lines.append(f"### Paths")
            for i, seq in enumerate(path_sequences[:5]):
                report_lines.append(f"{i+1}. {' -> '.join(seq)}")
            report_lines.append(f"")

    # Predictions
    if predictions:
        report_lines.append(f"## Oracle Predictions")
        for pred in predictions[:10]:
            report_lines.append(f"- **{pred.source} -> {pred.target}** (conf: {pred.confidence:.2f})")
            report_lines.append(f"  - Strategy: {pred.strategy_name}")
            report_lines.append(f"  - Reason: {pred.reasoning[:100]}...")
        report_lines.append(f"")

    report_content = "\n".join(report_lines)

    # Save report
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    else:
        report_path = Path("pipeline_report.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"  Report saved to: {report_path}")
    print()

    # =========================================================================
    # DONE
    # =========================================================================
    print("=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print()
    print("Results Summary:")
    print(f"  - Objects: {stats['objects']}")
    print(f"  - Morphisms: {stats['morphisms']}")
    if curvature_result:
        print(f"  - Curvature range: [{curvature_result.statistics['min']:.2f}, {curvature_result.statistics['max']:.2f}]")
    if decomposition:
        print(f"  - Geometric regions: {decomposition.num_regions}")
    if paths:
        print(f"  - Paths found: {len(paths)}")
    if predictions:
        print(f"  - Predictions: {len(predictions)}")
    print(f"  - Report: {report_path}")
    print()

    return {
        'store': store,
        'curvature': curvature_result,
        'decomposition': decomposition,
        'paths': paths,
        'homotopy': homotopy_result,
        'geo_homotopy': geo_homotopy_result,
        'predictions': predictions,
        'report_path': report_path
    }


def main():
    parser = argparse.ArgumentParser(
        description="KOMPOSOS-III Full Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with built-in physics dataset
  python pipeline.py --use-physics

  # Test with physics dataset and specific query
  python pipeline.py --use-physics --source Newton --target Schrodinger

  # Use your own data
  python pipeline.py --corpus ./my_corpus

  # Use existing database
  python pipeline.py --db ./store.db --source ConceptA --target ConceptB

  # Fast mode (skip embeddings)
  python pipeline.py --use-physics --skip-embeddings
"""
    )

    parser.add_argument("--corpus", help="Path to corpus directory")
    parser.add_argument("--db", help="Path to existing database file")
    parser.add_argument("--source", help="Source concept for path analysis")
    parser.add_argument("--target", help="Target concept for path analysis")
    parser.add_argument("--use-physics", action="store_true", help="Use built-in physics dataset")
    parser.add_argument("--output", help="Output directory for reports")
    parser.add_argument("--max-paths", type=int, default=10, help="Max paths to analyze")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding computation")

    args = parser.parse_args()

    if not args.corpus and not args.db and not args.use_physics:
        parser.print_help()
        print("\nError: Must specify --corpus, --db, or --use-physics")
        sys.exit(1)

    run_pipeline(
        corpus_path=args.corpus,
        db_path=args.db,
        source=args.source,
        target=args.target,
        use_physics=args.use_physics,
        output_dir=args.output,
        max_paths=args.max_paths,
        skip_embeddings=args.skip_embeddings
    )


if __name__ == "__main__":
    main()
