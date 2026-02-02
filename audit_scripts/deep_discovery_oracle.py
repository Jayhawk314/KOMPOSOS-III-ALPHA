"""
DEEP DISCOVERY ORACLE
=====================

Penalizes predictions that are "too easy" - derivable from simple graph traversal.

PRINCIPLE:
- Compositional predictions (2-hop, 3-hop) are expected from category theory
- True discovery requires finding interactions NOT in transitive closure
- Oracle should DOWN-WEIGHT shallow predictions, UP-WEIGHT deep discoveries

USAGE:
    from deep_discovery_oracle import DeepDiscoveryFilter

    filter = DeepDiscoveryFilter(db_path)
    predictions = oracle.predict(...)

    # Apply deep discovery penalty
    filtered = filter.apply_deep_discovery_penalty(predictions)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
from typing import List, Dict
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class PenaltyConfig:
    """Configuration for compositional penalties."""
    direct_penalty: float = 0.9         # 90% penalty for direct edges (already known)
    two_hop_penalty: float = 0.5        # 50% penalty for 2-hop (expected from composition)
    three_hop_penalty: float = 0.3      # 30% penalty for 3-hop (somewhat expected)
    truly_novel_bonus: float = 1.2      # 20% bonus for deep discoveries


class DeepDiscoveryFilter:
    """
    Filter that penalizes compositionally-derivable predictions.

    Integrates with existing CategoricalOracle to create "Deep Discovery Mode":
    - Predictions reachable via graph traversal get confidence penalty
    - Predictions NOT reachable get confidence boost
    - Final output prioritizes genuinely novel biology
    """

    def __init__(self, db_path: str, config: PenaltyConfig = None):
        self.db_path = db_path
        self.config = config or PenaltyConfig()
        self.graph = defaultdict(set)
        self._build_graph()

    def _build_graph(self):
        """Build directed graph from training data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT source_name, target_name FROM morphisms")
        edges = cursor.fetchall()
        conn.close()

        for source, target in edges:
            self.graph[source].add(target)

    def find_shortest_path(self, source: str, target: str, max_hops: int = 3):
        """BFS shortest path (returns None if not reachable)."""
        if source == target:
            return [source]

        if source not in self.graph:
            return None

        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            if len(path) > max_hops:
                continue

            for neighbor in self.graph.get(current, []):
                if neighbor == target:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def classify_prediction(self, source: str, target: str) -> Dict:
        """
        Classify prediction as DIRECT, 2-HOP, 3-HOP, or TRULY_NOVEL.

        Returns:
            {
                'classification': str,
                'hops': int or None,
                'penalty_factor': float (multiply confidence by this)
            }
        """
        # Check direct edge (in training)
        if target in self.graph.get(source, []):
            return {
                'classification': 'DIRECT',
                'hops': 1,
                'penalty_factor': self.config.direct_penalty,
                'is_deep_discovery': False
            }

        # Check reverse (bidirectional)
        if source in self.graph.get(target, []):
            return {
                'classification': 'DIRECT_REVERSE',
                'hops': 1,
                'penalty_factor': self.config.direct_penalty,
                'is_deep_discovery': False
            }

        # Check compositional (2-hop, 3-hop)
        path = self.find_shortest_path(source, target, max_hops=3)

        if path:
            hops = len(path) - 1
            if hops == 2:
                penalty = self.config.two_hop_penalty
            elif hops == 3:
                penalty = self.config.three_hop_penalty
            else:
                penalty = 0.9  # Higher hops still penalized

            return {
                'classification': f'{hops}-HOP',
                'hops': hops,
                'penalty_factor': penalty,
                'is_deep_discovery': False,
                'path': path
            }

        # Check reverse path
        reverse_path = self.find_shortest_path(target, source, max_hops=3)
        if reverse_path:
            hops = len(reverse_path) - 1
            return {
                'classification': f'{hops}-HOP_REVERSE',
                'hops': hops,
                'penalty_factor': self.config.three_hop_penalty,  # Reverse paths less reliable
                'is_deep_discovery': False,
                'path': reverse_path
            }

        # TRULY NOVEL - not reachable via composition
        return {
            'classification': 'TRULY_NOVEL',
            'hops': None,
            'penalty_factor': self.config.truly_novel_bonus,
            'is_deep_discovery': True
        }

    def apply_deep_discovery_penalty(self, predictions: List[Dict]) -> List[Dict]:
        """
        Apply deep discovery penalty to list of predictions.

        Args:
            predictions: List of dicts with keys: source, target, confidence

        Returns:
            Same list with added fields:
            - original_confidence
            - adjusted_confidence
            - classification
            - is_deep_discovery
            - penalty_factor
        """
        results = []

        for pred in predictions:
            source = pred['source']
            target = pred['target']
            confidence = pred['confidence']

            # Classify
            classification = self.classify_prediction(source, target)

            # Apply penalty
            adjusted_confidence = confidence * classification['penalty_factor']

            results.append({
                **pred,
                'original_confidence': confidence,
                'adjusted_confidence': adjusted_confidence,
                'classification': classification['classification'],
                'is_deep_discovery': classification['is_deep_discovery'],
                'penalty_factor': classification['penalty_factor'],
                'hops': classification.get('hops'),
                'path': ' -> '.join(classification['path']) if 'path' in classification else None
            })

        # Re-sort by adjusted confidence
        results.sort(key=lambda x: x['adjusted_confidence'], reverse=True)

        return results

    def get_deep_discoveries(self, predictions: List[Dict], min_confidence: float = 0.5) -> List[Dict]:
        """
        Extract only TRULY_NOVEL predictions above threshold.

        This is the "Deep Discovery" output - predictions that:
        1. Are not in training data (direct)
        2. Are not reachable via 2-hop or 3-hop composition
        3. Have confidence >= min_confidence (after penalty/bonus)

        Returns:
            Filtered list of deep discoveries, sorted by adjusted confidence
        """
        penalized = self.apply_deep_discovery_penalty(predictions)

        deep = [
            p for p in penalized
            if p['is_deep_discovery'] and p['adjusted_confidence'] >= min_confidence
        ]

        return deep

    def print_comparison_report(self, predictions: List[Dict]):
        """
        Print before/after report showing impact of deep discovery filter.
        """
        penalized = self.apply_deep_discovery_penalty(predictions)

        # Statistics
        total = len(penalized)
        deep = sum(1 for p in penalized if p['is_deep_discovery'])

        classifications = defaultdict(int)
        for p in penalized:
            classifications[p['classification']] += 1

        print()
        print("=" * 80)
        print("DEEP DISCOVERY FILTER REPORT")
        print("=" * 80)
        print()

        print(f"Total predictions: {total}")
        print(f"Deep discoveries:  {deep} ({deep/total*100:.1f}%)")
        print(f"Shallow (compositional): {total - deep} ({(total-deep)/total*100:.1f}%)")
        print()

        print("Classification breakdown:")
        for cls, count in sorted(classifications.items()):
            print(f"  {cls:20s}: {count:3d} ({count/total*100:5.1f}%)")

        print()
        print("=" * 80)
        print("TOP 10 PREDICTIONS (BEFORE PENALTY)")
        print("=" * 80)
        print()

        original_top10 = sorted(predictions, key=lambda x: x['confidence'], reverse=True)[:10]
        for i, p in enumerate(original_top10, 1):
            print(f"{i:2d}. {p['source']:10s} -> {p['target']:10s} | conf={p['confidence']:.3f}")

        print()
        print("=" * 80)
        print("TOP 10 PREDICTIONS (AFTER PENALTY)")
        print("=" * 80)
        print()

        adjusted_top10 = penalized[:10]
        for i, p in enumerate(adjusted_top10, 1):
            marker = "[DEEP]" if p['is_deep_discovery'] else f"[{p['classification']}]"
            print(f"{i:2d}. {p['source']:10s} -> {p['target']:10s} | "
                  f"conf={p['adjusted_confidence']:.3f} | {marker:15s}")

        print()

        # Show examples of penalized predictions
        print("=" * 80)
        print("PENALIZED PREDICTIONS (Was top 10, now demoted)")
        print("=" * 80)
        print()

        original_top10_pairs = {(p['source'], p['target']) for p in original_top10}
        adjusted_top10_pairs = {(p['source'], p['target']) for p in adjusted_top10}

        demoted = original_top10_pairs - adjusted_top10_pairs

        if demoted:
            for pair in demoted:
                pred = next((p for p in penalized if (p['source'], p['target']) == pair), None)
                if pred:
                    print(f"{pred['source']:10s} -> {pred['target']:10s} | "
                          f"orig={pred['original_confidence']:.3f} -> adj={pred['adjusted_confidence']:.3f} | "
                          f"{pred['classification']}")
                    if pred.get('path'):
                        print(f"  Path: {pred['path']}")
        else:
            print("(None - top 10 unchanged)")

        print()

        # Show examples of promoted predictions
        print("=" * 80)
        print("PROMOTED PREDICTIONS (Deep discoveries boosted to top 10)")
        print("=" * 80)
        print()

        promoted = adjusted_top10_pairs - original_top10_pairs

        if promoted:
            for pair in promoted:
                pred = next((p for p in penalized if (p['source'], p['target']) == pair), None)
                if pred:
                    print(f"{pred['source']:10s} -> {pred['target']:10s} | "
                          f"orig={pred['original_confidence']:.3f} -> adj={pred['adjusted_confidence']:.3f} | "
                          f"{pred['classification']}")
        else:
            print("(None - no deep discoveries promoted)")

        print()


# Integration example
def integrate_with_oracle_example():
    """
    Example of how to integrate with existing CategoricalOracle.
    """
    print("""
    INTEGRATION EXAMPLE
    ===================

    # In your main pipeline:

    from oracle import CategoricalOracle
    from oracle.conjecture import ConjectureEngine
    from audit_scripts.deep_discovery_oracle import DeepDiscoveryFilter

    # Standard oracle
    oracle = CategoricalOracle(store, embeddings)
    engine = ConjectureEngine(oracle)
    result = engine.conjecture(top_k=100)

    # Convert to dict format
    predictions = [
        {
            'source': c.source,
            'target': c.target,
            'confidence': c.top_confidence,
            'relation': c.best.predicted_relation if c.best else 'unknown'
        }
        for c in result.conjectures
    ]

    # Apply deep discovery filter
    filter = DeepDiscoveryFilter('data/proteins/cancer_proteins.db')
    deep_discoveries = filter.get_deep_discoveries(predictions, min_confidence=0.5)

    # Report
    print(f"Original predictions: {len(predictions)}")
    print(f"Deep discoveries: {len(deep_discoveries)}")
    print()
    print("Top 10 deep discoveries:")
    for i, pred in enumerate(deep_discoveries[:10], 1):
        print(f"{i}. {pred['source']} -> {pred['target']} (conf={pred['adjusted_confidence']:.3f})")
    """)


if __name__ == "__main__":
    integrate_with_oracle_example()
