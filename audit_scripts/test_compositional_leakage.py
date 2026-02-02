"""
COMPOSITIONAL LEAKAGE DETECTOR
===============================

Tests if "novel" predictions are actually reachable via graph composition
from the training data. This is the smoking gun test for data leakage.

AUDIT HYPOTHESIS:
- System claims 93% novelty (not in 55-edge training set)
- But predictions might be reachable via 2-hop or 3-hop paths
- If true: "novelty" is an artifact, not discovery

OUTPUT:
1. Classification of each prediction: DIRECT, 2-HOP, 3-HOP, TRULY_NOVEL
2. Corrected novelty metric (excludes compositional predictions)
3. Deep Discovery Rate (predictions NOT derivable from graph)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import json
import csv
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque


class CompositionalLeakageDetector:
    """Detect if predictions are reachable via graph traversal."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.graph = defaultdict(set)  # source -> {targets}
        self.edge_details = {}  # (source, target) -> morphism info
        self._build_graph()

    def _build_graph(self):
        """Build directed graph from morphisms table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT source_name, target_name, name, confidence FROM morphisms")
        edges = cursor.fetchall()
        conn.close()

        for source, target, relation, confidence in edges:
            self.graph[source].add(target)
            self.edge_details[(source, target)] = {
                'relation': relation,
                'confidence': confidence
            }

        print(f"[Graph] Loaded {len(edges)} edges, {len(self.graph)} nodes")

    def find_shortest_path(self, source: str, target: str, max_hops: int = 3) -> Optional[List[str]]:
        """
        BFS to find shortest path from source to target.

        Returns:
            List of nodes in path, or None if not reachable
        """
        if source == target:
            return [source]

        if source not in self.graph:
            return None

        # BFS with path tracking
        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            # Max hops reached
            if len(path) > max_hops:
                continue

            # Explore neighbors
            for neighbor in self.graph.get(current, []):
                if neighbor == target:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def classify_prediction(self, source: str, target: str) -> Dict:
        """
        Classify a prediction based on graph reachability.

        Returns:
            {
                'classification': 'DIRECT' | '2-HOP' | '3-HOP' | 'TRULY_NOVEL',
                'path': [nodes] or None,
                'hops': int or None,
                'is_leakage': bool,
                'leakage_type': str or None
            }
        """
        # Check direct edge
        if (source, target) in self.edge_details:
            return {
                'classification': 'DIRECT',
                'path': [source, target],
                'hops': 1,
                'is_leakage': True,
                'leakage_type': 'IN_TRAINING_DIRECT',
                'details': self.edge_details[(source, target)]
            }

        # Check reverse edge (bidirectional interactions)
        if (target, source) in self.edge_details:
            return {
                'classification': 'DIRECT_REVERSE',
                'path': [target, source],
                'hops': 1,
                'is_leakage': True,
                'leakage_type': 'IN_TRAINING_REVERSE',
                'details': self.edge_details[(target, source)]
            }

        # Check 2-hop path
        path = self.find_shortest_path(source, target, max_hops=2)
        if path and len(path) == 3:
            return {
                'classification': '2-HOP',
                'path': path,
                'hops': 2,
                'is_leakage': True,
                'leakage_type': 'COMPOSITIONAL_2HOP',
                'intermediate': path[1]
            }

        # Check 3-hop path
        path = self.find_shortest_path(source, target, max_hops=3)
        if path and len(path) == 4:
            return {
                'classification': '3-HOP',
                'path': path,
                'hops': 3,
                'is_leakage': True,
                'leakage_type': 'COMPOSITIONAL_3HOP',
                'intermediates': path[1:-1]
            }

        # Check reverse path (target -> source)
        reverse_path = self.find_shortest_path(target, source, max_hops=3)
        if reverse_path:
            return {
                'classification': f'{len(reverse_path)-1}-HOP_REVERSE',
                'path': reverse_path,
                'hops': len(reverse_path) - 1,
                'is_leakage': True,
                'leakage_type': f'COMPOSITIONAL_{len(reverse_path)-1}HOP_REVERSE'
            }

        # Truly novel - not reachable
        return {
            'classification': 'TRULY_NOVEL',
            'path': None,
            'hops': None,
            'is_leakage': False,
            'leakage_type': None
        }

    def audit_predictions(self, predictions_csv: str) -> Dict:
        """
        Audit all predictions for compositional leakage.

        Args:
            predictions_csv: CSV with columns: source, target, confidence, system

        Returns:
            Audit report with statistics
        """
        print("=" * 80)
        print("COMPOSITIONAL LEAKAGE AUDIT")
        print("=" * 80)
        print()

        # Load predictions
        predictions = []
        with open(predictions_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                predictions.append({
                    'source': row['source'],
                    'target': row['target'],
                    'confidence': float(row['confidence']),
                    'system': row['system']
                })

        print(f"Loaded {len(predictions)} predictions")
        print()

        # Classify each prediction
        results = []
        leakage_counts = defaultdict(int)

        for pred in predictions:
            classification = self.classify_prediction(pred['source'], pred['target'])

            result = {
                **pred,
                **classification
            }
            results.append(result)

            leakage_counts[classification['classification']] += 1

        # Statistics
        total = len(results)
        leaked = sum(1 for r in results if r['is_leakage'])
        truly_novel = sum(1 for r in results if r['classification'] == 'TRULY_NOVEL')

        # Original "novelty" (not in direct edges)
        direct = sum(1 for r in results if r['classification'] in ['DIRECT', 'DIRECT_REVERSE'])
        original_novelty = total - direct

        # Corrected novelty (not reachable via composition)
        corrected_novelty = truly_novel

        print("=" * 80)
        print("LEAKAGE CLASSIFICATION")
        print("=" * 80)
        print()

        for classification, count in sorted(leakage_counts.items()):
            percentage = count / total * 100
            print(f"{classification:20s}: {count:3d} ({percentage:5.1f}%)")

        print()
        print("=" * 80)
        print("NOVELTY METRICS")
        print("=" * 80)
        print()

        print(f"Total predictions:              {total}")
        print(f"In training (direct):           {direct} ({direct/total*100:.1f}%)")
        print(f"Original novelty metric:        {original_novelty} ({original_novelty/total*100:.1f}%)")
        print()
        print(f"CORRECTED METRICS:")
        print(f"Leaked via composition:         {leaked - direct} ({(leaked-direct)/total*100:.1f}%)")
        print(f"Truly novel (not reachable):    {corrected_novelty} ({corrected_novelty/total*100:.1f}%)")
        print()
        print(f"DEEP DISCOVERY RATE:            {corrected_novelty/total*100:.1f}%")
        print(f"  (predictions NOT derivable from graph)")
        print()

        # Breakdown by system
        print("=" * 80)
        print("BY SYSTEM")
        print("=" * 80)
        print()

        systems = set(r['system'] for r in results)
        for system in sorted(systems):
            system_results = [r for r in results if r['system'] == system]
            system_total = len(system_results)
            system_novel = sum(1 for r in system_results if r['classification'] == 'TRULY_NOVEL')

            print(f"{system} System:")
            print(f"  Total predictions:     {system_total}")
            print(f"  Truly novel:           {system_novel} ({system_novel/system_total*100:.1f}%)")
            print()

        # Examples of leakage
        print("=" * 80)
        print("LEAKAGE EXAMPLES (2-HOP PATHS)")
        print("=" * 80)
        print()

        two_hop = [r for r in results if r['classification'] == '2-HOP'][:10]
        for r in two_hop:
            path_str = " -> ".join(r['path'])
            print(f"{r['source']:10s} -> {r['target']:10s} | Path: {path_str}")

        print()

        # Examples of deep discovery
        print("=" * 80)
        print("DEEP DISCOVERIES (Truly Novel)")
        print("=" * 80)
        print()

        deep = [r for r in results if r['classification'] == 'TRULY_NOVEL'][:20]
        if deep:
            print(f"Found {len([r for r in results if r['classification'] == 'TRULY_NOVEL'])} truly novel predictions")
            print()
            print("Top 20 by confidence:")
            deep_sorted = sorted(deep, key=lambda x: x['confidence'], reverse=True)
            for i, r in enumerate(deep_sorted[:20], 1):
                print(f"{i:2d}. {r['source']:10s} -> {r['target']:10s} | conf={r['confidence']:.3f} | {r['system']}")
        else:
            print("WARNING: NO TRULY NOVEL PREDICTIONS FOUND")
            print("All predictions are derivable from training graph!")

        print()

        # Save detailed results
        output_file = Path(predictions_csv).parent / "leakage_audit_results.csv"
        with open(output_file, 'w', newline='') as f:
            fieldnames = ['source', 'target', 'confidence', 'system',
                         'classification', 'hops', 'is_leakage', 'leakage_type', 'path']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                writer.writerow({
                    'source': r['source'],
                    'target': r['target'],
                    'confidence': r['confidence'],
                    'system': r['system'],
                    'classification': r['classification'],
                    'hops': r['hops'],
                    'is_leakage': r['is_leakage'],
                    'leakage_type': r['leakage_type'],
                    'path': ' -> '.join(r['path']) if r['path'] else 'None'
                })

        print(f"Detailed results saved: {output_file}")
        print()

        # Return summary
        return {
            'total_predictions': total,
            'direct_in_training': direct,
            'original_novelty': original_novelty,
            'original_novelty_rate': original_novelty / total,
            'compositional_leakage': leaked - direct,
            'truly_novel': corrected_novelty,
            'deep_discovery_rate': corrected_novelty / total,
            'classification_breakdown': dict(leakage_counts),
            'results': results
        }


def test_validation_set_leakage(detector: CompositionalLeakageDetector):
    """
    SMOKING GUN TEST: Check if validation pairs are reachable in training graph.

    This proves whether the 6% precision is from discovery or graph traversal.
    """
    print("=" * 80)
    print("VALIDATION SET LEAKAGE TEST")
    print("=" * 80)
    print()
    print("Testing if KNOWN_VALIDATIONS pairs are reachable in training graph...")
    print()

    # Validation pairs from validate_biological_embeddings.py
    validation_pairs = [
        ("KRAS", "MYC"),
        ("EGFR", "MYC"),
        ("PTEN", "BAX"),
        ("EGFR", "BRAF"),
        ("EGFR", "RAF1"),
        ("BRCA1", "RAD51"),
        ("PIK3CA", "MYC"),
        ("NRAS", "MYC"),
        ("STAT3", "KRAS"),
        ("RAF1", "TP53"),
        ("BRAF", "TP53"),
        ("CDK4", "TP53"),
        ("CDK6", "TP53"),
    ]

    results = []
    for source, target in validation_pairs:
        classification = detector.classify_prediction(source, target)
        results.append({
            'pair': f"{source} -> {target}",
            **classification
        })

    # Statistics
    leaked = sum(1 for r in results if r['is_leakage'])

    print(f"Total validation pairs:  {len(results)}")
    print(f"Reachable via graph:     {leaked} ({leaked/len(results)*100:.1f}%)")
    print(f"Truly independent:       {len(results) - leaked} ({(len(results)-leaked)/len(results)*100:.1f}%)")
    print()

    print("BREAKDOWN:")
    print()
    for r in results:
        status = "LEAKED" if r['is_leakage'] else "CLEAN"
        path_str = " -> ".join(r['path']) if r['path'] else "None"
        print(f"{status:8s} | {r['pair']:25s} | {r['classification']:15s} | {path_str}")

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()

    if leaked > len(results) * 0.5:
        print("WARNING: MAJORITY OF VALIDATION SET IS LEAKED")
        print()
        print(f"{leaked}/{len(results)} validation pairs are reachable via graph composition.")
        print("This means the 6% precision is NOT from true discovery.")
        print("The system is finding paths in the training graph, not novel biology.")
        print()
        print("RECOMMENDATION:")
        print("- Use truly independent validation set (not compositionally reachable)")
        print("- Report 'Deep Discovery' precision separately")
    elif leaked > 0:
        print(f"PARTIAL LEAKAGE: {leaked}/{len(results)} validation pairs are reachable")
        print()
        print("Some validation pairs are compositionally derivable from training data.")
        print("Precision metric should exclude these for fair evaluation.")
    else:
        print("VALIDATION SET IS CLEAN")
        print()
        print("No compositional leakage detected. Validation is independent.")

    print()

    return results


def main():
    """Run all leakage tests."""

    # Initialize detector
    db_path = "data/proteins/cancer_proteins.db"
    detector = CompositionalLeakageDetector(db_path)

    print()
    print("=" * 80)
    print("KOMPOSOS-III COMPOSITIONAL LEAKAGE AUDIT")
    print("=" * 80)
    print()

    # Test 1: Check if predictions CSV exists
    predictions_file = Path("reports/predictions_with_novelty.csv")

    if not predictions_file.exists():
        print(f"Predictions file not found: {predictions_file}")
        print("Run scripts/check_novelty_comprehensive.py first to generate it")
        print()
        print("Proceeding with validation set test only...")
        print()

        # Run validation set test
        test_validation_set_leakage(detector)

    else:
        # Run full audit on predictions
        audit_report = detector.audit_predictions(str(predictions_file))

        print()

        # Run validation set test
        test_validation_set_leakage(detector)

        # Save summary
        summary_file = Path("reports/leakage_audit_summary.json")
        with open(summary_file, 'w') as f:
            # Remove results list (too large for JSON)
            summary = {k: v for k, v in audit_report.items() if k != 'results'}
            json.dump(summary, f, indent=2)

        print(f"Summary saved: {summary_file}")

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
