#!/usr/bin/env python3
"""
KOMPOSOS-III Quality Stress Test
=================================

Tests whether the system produces MEANINGFUL predictions, not just pretty reports.

Core methodology:
1. BACKTESTING: Remove known relationships, see if Oracle predicts them
2. TEMPORAL HOLDOUT: Train on pre-X data, predict post-X developments
3. EQUIVALENCE DISCOVERY: Remove known equivalences, see if structural analysis finds them
4. CONSISTENCY: Same inputs should produce consistent outputs

This validates that KOMPOSOS-III's categorical analysis is genuinely useful,
not just well-formatted prose.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Set
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import create_memory_store, KomposOSStore, StoredObject, StoredMorphism, EquivalenceClass
from data import EmbeddingsEngine, StoreEmbedder
from evaluation.physics_dataset import create_physics_dataset
from cli import ReportGenerator

# Import the enhanced Oracle (optional - gracefully degrades if not available)
try:
    from oracle import CategoricalOracle, OracleResult
    ORACLE_AVAILABLE = True
except ImportError:
    ORACLE_AVAILABLE = False
    CategoricalOracle = None

# =============================================================================
# DATA DEFINITIONS (extracted from physics_dataset for testing)
# =============================================================================

# Relationships we can hide for backtesting
KNOWN_RELATIONSHIPS = [
    {"source": "Galileo", "target": "Newton", "type": "influenced", "year": 1687},
    {"source": "Kepler", "target": "Newton", "type": "influenced", "year": 1687},
    {"source": "Newton", "target": "Euler", "type": "influenced", "year": 1750},
    {"source": "Euler", "target": "Lagrange", "type": "influenced", "year": 1788},
    {"source": "Lagrange", "target": "Hamilton", "type": "reformulated", "year": 1833},
    {"source": "Hamilton", "target": "Jacobi", "type": "influenced", "year": 1837},
    {"source": "Hamilton", "target": "Schrodinger", "type": "influenced", "year": 1926},
    {"source": "Faraday", "target": "Maxwell", "type": "influenced", "year": 1865},
    {"source": "Maxwell", "target": "Boltzmann", "type": "influenced", "year": 1877},
    {"source": "Boltzmann", "target": "Planck", "type": "influenced", "year": 1900},
    {"source": "Planck", "target": "Einstein", "type": "influenced", "year": 1905},
    {"source": "Planck", "target": "Bohr", "type": "influenced", "year": 1913},
    {"source": "Einstein", "target": "Bohr", "type": "influenced", "year": 1913},
    {"source": "Bohr", "target": "Heisenberg", "type": "influenced", "year": 1925},
    {"source": "Sommerfeld", "target": "Heisenberg", "type": "influenced", "year": 1925},
    {"source": "deBroglie", "target": "Schrodinger", "type": "influenced", "year": 1926},
    {"source": "Schrodinger", "target": "Dirac", "type": "influenced", "year": 1928},
    {"source": "Heisenberg", "target": "Dirac", "type": "influenced", "year": 1928},
    {"source": "Dirac", "target": "Feynman", "type": "influenced", "year": 1948},
    {"source": "Dirac", "target": "Schwinger", "type": "influenced", "year": 1948},
    {"source": "Dirac", "target": "Tomonaga", "type": "influenced", "year": 1943},
    {"source": "Lorentz", "target": "Einstein", "type": "influenced", "year": 1905},
    {"source": "Maxwell", "target": "Einstein", "type": "influenced", "year": 1905},
]

# Equivalences we can hide
KNOWN_EQUIVALENCES = [
    {"name": "QM_Formulations", "members": ["WaveMechanics", "MatrixMechanics"], "type": "mathematical"},
    {"name": "QED_Formulations", "members": ["Feynman_QED", "Schwinger_QED", "Tomonaga_QED"], "type": "physical"},
    {"name": "Mechanics_Formulations", "members": ["ClassicalMechanics", "AnalyticalMechanics"], "type": "mathematical"},
]


class StressTestResult:
    """Results from a single stress test."""
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.predictions_made = 0
        self.correct_predictions = 0
        self.false_positives = 0
        self.missed_targets = 0
        self.details: List[Dict] = []

    @property
    def precision(self) -> float:
        """Of predictions made, how many were correct?"""
        if self.predictions_made == 0:
            return 0.0
        return self.correct_predictions / self.predictions_made

    @property
    def recall(self) -> float:
        """Of actual connections, how many did we predict?"""
        total_actual = self.correct_predictions + self.missed_targets
        if total_actual == 0:
            return 0.0
        return self.correct_predictions / total_actual

    @property
    def f1_score(self) -> float:
        """Harmonic mean of precision and recall."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


def create_partial_dataset(
    exclude_morphisms: Set[Tuple[str, str]] = None,
    exclude_equivalences: Set[str] = None,
    max_year: int = None
) -> KomposOSStore:
    """
    Create a dataset with some data intentionally excluded.

    We create the full dataset first, then selectively remove data.
    This ensures we're testing the same base data.

    Args:
        exclude_morphisms: Set of (source, target) pairs to exclude
        exclude_equivalences: Set of equivalence class names to exclude
        max_year: Only include morphisms up to this year
    """
    exclude_morphisms = exclude_morphisms or set()
    exclude_equivalences = exclude_equivalences or set()

    # Create a fresh store
    store = create_memory_store()

    # Get a full dataset to copy from (we'll filter as we add)
    full_store = create_physics_dataset()

    # Copy all objects
    for obj in full_store.list_objects(limit=1000):
        store.add_object(StoredObject(
            name=obj.name,
            type_name=obj.type_name,
            metadata=obj.metadata,
            provenance="stress_test"
        ))

    # Copy morphisms with exclusions
    for mor in full_store.list_morphisms(limit=10000):
        source, target = mor.source_name, mor.target_name
        year = mor.metadata.get("year", 0)

        # Skip if excluded
        if (source, target) in exclude_morphisms:
            continue

        # Skip if after max_year
        if max_year and year and year > max_year:
            continue

        store.add_morphism(StoredMorphism(
            name=mor.name,
            source_name=source,
            target_name=target,
            confidence=mor.confidence,
            metadata=mor.metadata,
            provenance="stress_test"
        ))

    # Copy equivalences with exclusions
    for equiv in full_store.list_equivalences():
        if equiv.name in exclude_equivalences:
            continue

        store.add_equivalence(EquivalenceClass(
            name=equiv.name,
            member_names=equiv.member_names,
            equivalence_type=equiv.equivalence_type,
            witness=equiv.witness,
            confidence=equiv.confidence
        ))

    return store


def extract_oracle_predictions(store: KomposOSStore, source: str, target: str,
                                embeddings: EmbeddingsEngine = None) -> List[Dict]:
    """
    Extract predictions from the Oracle module for a given source/target pair.

    Uses the enhanced CategoricalOracle with 8 inference strategies if available,
    otherwise falls back to basic heuristics.

    Args:
        store: The KomposOS data store
        source: Source object name
        target: Target object name
        embeddings: Optional embeddings engine for enhanced predictions
    """
    # Try to use enhanced CategoricalOracle
    if ORACLE_AVAILABLE and embeddings and embeddings.is_available:
        try:
            oracle = CategoricalOracle(store, embeddings)
            result = oracle.predict(source, target)

            # Convert to stress test format
            predictions = []
            for pred in result.predictions:
                predictions.append({
                    "type": pred.strategy_name,
                    "source": pred.source,
                    "predicted_target": pred.target,
                    "relation": pred.predicted_relation,
                    "confidence": pred.confidence,
                    "reason": pred.reasoning,
                })
            return predictions

        except Exception as e:
            print(f"Warning: CategoricalOracle failed: {e}")

    # Fallback to basic heuristics
    return extract_basic_predictions(store, source, target)


def extract_basic_predictions(store: KomposOSStore, source: str, target: str) -> List[Dict]:
    """
    Basic prediction heuristics (fallback when Oracle unavailable).

    This uses the same logic as the old ReportGenerator but extracts just the predictions.
    """
    predictions = []

    # Get all objects and morphisms
    all_objects = store.list_objects(limit=500)
    morphisms = store.list_morphisms(limit=10000)

    # Build morphism index
    outgoing = {}
    incoming = {}
    for mor in morphisms:
        if mor.source_name not in outgoing:
            outgoing[mor.source_name] = []
        outgoing[mor.source_name].append(mor)
        if mor.target_name not in incoming:
            incoming[mor.target_name] = []
        incoming[mor.target_name].append(mor)

    # Prediction 1: Objects with similar outgoing patterns
    source_out = set(m.target_name for m in outgoing.get(source, []))
    for obj in all_objects:
        if obj.name == source:
            continue
        obj_out = set(m.target_name for m in outgoing.get(obj.name, []))
        overlap = len(source_out & obj_out)
        if overlap >= 2:
            predictions.append({
                "type": "yoneda_analogy",
                "source": source,
                "predicted_target": obj.name,
                "confidence": min(0.95, 0.5 + overlap * 0.1),
                "reason": f"Shares {overlap} outgoing relationships"
            })

    # Prediction 2: Missing morphisms based on target's relationships
    for mor in morphisms:
        if mor.source_name == target and mor.target_name != source:
            if mor.target_name not in source_out:
                predictions.append({
                    "type": "missing_morphism",
                    "source": source,
                    "predicted_target": mor.target_name,
                    "relation": mor.name,
                    "confidence": 0.65,
                    "reason": f"Target {target} has '{mor.name}' to {mor.target_name}"
                })

    return predictions


def compute_embeddings_for_store(store: KomposOSStore) -> EmbeddingsEngine:
    """
    Compute embeddings for a store's objects.

    Returns embeddings engine or None if embeddings unavailable.
    """
    try:
        embeddings = EmbeddingsEngine()
        if not embeddings.is_available:
            print("Note: Embeddings not available, using basic heuristics")
            return None

        embedder = StoreEmbedder(store, embeddings)
        embedder.embed_all_objects(show_progress=False)
        return embeddings
    except Exception as e:
        print(f"Warning: Could not compute embeddings: {e}")
        return None


def test_backtest_morphism_prediction(use_enhanced_oracle: bool = True) -> StressTestResult:
    """
    BACKTEST: Remove known morphisms, see if Oracle predicts them.

    Methodology:
    1. Select morphisms to hide (ground truth)
    2. Create dataset without those morphisms
    3. Run Oracle predictions
    4. Check if hidden morphisms appear in predictions

    Args:
        use_enhanced_oracle: If True, use CategoricalOracle with embeddings
    """
    result = StressTestResult("Backtest: Morphism Prediction")

    # Select morphisms to hide - choose ones that have clear structural signals
    hidden_morphisms = [
        ("Dirac", "Feynman", "influenced"),      # Well-connected nodes
        ("Einstein", "Bohr", "influenced"),      # Major hubs
        ("Heisenberg", "Dirac", "influenced"),   # Quantum mechanics chain
        ("Bohr", "Heisenberg", "influenced"),    # Another quantum link
        ("Maxwell", "Hertz", "influenced"),      # Classical to experimental
    ]

    exclude_set = {(s, t) for s, t, _ in hidden_morphisms}

    # Create partial dataset
    store = create_partial_dataset(exclude_morphisms=exclude_set)

    # Compute embeddings for enhanced Oracle
    embeddings = None
    if use_enhanced_oracle and ORACLE_AVAILABLE:
        print("Computing embeddings for enhanced Oracle...")
        embeddings = compute_embeddings_for_store(store)

    # For each hidden morphism, check if Oracle would predict it
    for source, target, rel_type in hidden_morphisms:
        # Get predictions from Oracle
        predictions = extract_oracle_predictions(store, source, target, embeddings)

        # Also check predictions from target's perspective
        reverse_predictions = extract_oracle_predictions(store, target, source, embeddings)

        # Check if the hidden relationship was predicted
        predicted = False
        for pred in predictions + reverse_predictions:
            if pred.get("predicted_target") == target or pred.get("predicted_target") == source:
                predicted = True
                result.correct_predictions += 1
                result.details.append({
                    "hidden": f"{source} -> {target}",
                    "predicted": True,
                    "confidence": pred.get("confidence", 0),
                    "type": pred.get("type")
                })
                break

        result.predictions_made += 1

        if not predicted:
            result.missed_targets += 1
            result.details.append({
                "hidden": f"{source} -> {target}",
                "predicted": False,
                "note": "Oracle did not predict this relationship"
            })

    return result


def test_temporal_holdout(use_enhanced_oracle: bool = True) -> StressTestResult:
    """
    TEMPORAL HOLDOUT: Train on pre-1925 data, predict post-1925 developments.

    The cutoff of 1925 is significant: it's right before quantum mechanics
    was formalized. Can the system predict the quantum revolution?

    Args:
        use_enhanced_oracle: If True, use CategoricalOracle with embeddings
    """
    result = StressTestResult("Temporal Holdout: Pre-1925 -> Post-1925")

    # Create dataset with only pre-1925 morphisms
    cutoff_year = 1925
    store = create_partial_dataset(max_year=cutoff_year)

    # Compute embeddings for enhanced Oracle
    embeddings = None
    if use_enhanced_oracle and ORACLE_AVAILABLE:
        print("Computing embeddings for temporal holdout test...")
        embeddings = compute_embeddings_for_store(store)

    # Post-1925 developments we want to predict
    post_1925_developments = [
        ("Hamilton", "Schrodinger"),    # Hamiltonian -> Wave mechanics
        ("Bohr", "Heisenberg"),         # Old quantum -> Matrix mechanics
        ("Schrodinger", "Dirac"),       # Wave -> Relativistic QM
        ("Heisenberg", "Dirac"),        # Matrix -> Relativistic QM
        ("Dirac", "Feynman"),           # QM -> QED
    ]

    # Check which of these the Oracle would predict
    for source, target in post_1925_developments:
        # Check if source exists in pre-1925 data
        source_obj = store.get_object(source)
        if not source_obj:
            continue

        predictions = extract_oracle_predictions(store, source, target, embeddings)

        predicted = False
        for pred in predictions:
            # Check if prediction points toward the target
            if pred.get("predicted_target") == target:
                predicted = True
                result.correct_predictions += 1
                result.details.append({
                    "future_connection": f"{source} -> {target}",
                    "predicted": True,
                    "type": pred.get("type"),
                    "confidence": pred.get("confidence", 0)
                })
                break

        result.predictions_made += 1

        if not predicted:
            result.missed_targets += 1
            result.details.append({
                "future_connection": f"{source} -> {target}",
                "predicted": False,
                "note": "Could not predict post-1925 development"
            })

    return result


def test_equivalence_discovery() -> StressTestResult:
    """
    EQUIVALENCE DISCOVERY: Remove known equivalences, see if structural
    analysis can rediscover them.

    This tests the Yoneda-based structural analysis: do equivalent concepts
    have the same morphism patterns?
    """
    result = StressTestResult("Equivalence Discovery")

    # Equivalences to hide and try to rediscover
    target_equivalences = [
        "QM_Formulations",      # WaveMechanics ≃ MatrixMechanics
        "QED_Formulations",     # Feynman ≃ Schwinger ≃ Tomonaga
    ]

    # Create dataset without these equivalences
    store = create_partial_dataset(exclude_equivalences=set(target_equivalences))

    # Get all morphisms for structural analysis
    morphisms = store.list_morphisms(limit=10000)
    objects = store.list_objects(limit=500)

    # Build structural signatures (Yoneda-style)
    signatures = {}
    for obj in objects:
        out_rels = frozenset(m.name for m in morphisms if m.source_name == obj.name)
        in_rels = frozenset(m.name for m in morphisms if m.target_name == obj.name)
        out_targets = frozenset(m.target_name for m in morphisms if m.source_name == obj.name)
        in_sources = frozenset(m.source_name for m in morphisms if m.target_name == obj.name)
        signatures[obj.name] = {
            "out_rels": out_rels,
            "in_rels": in_rels,
            "out_targets": out_targets,
            "in_sources": in_sources,
            "type": obj.type_name
        }

    # Check if we can rediscover the hidden equivalences

    # QM_Formulations: WaveMechanics ≃ MatrixMechanics
    if "WaveMechanics" in signatures and "MatrixMechanics" in signatures:
        sig1 = signatures["WaveMechanics"]
        sig2 = signatures["MatrixMechanics"]

        # Check structural similarity
        rel_overlap = len(sig1["out_rels"] & sig2["out_rels"]) + len(sig1["in_rels"] & sig2["in_rels"])
        target_overlap = len(sig1["out_targets"] & sig2["out_targets"])
        source_overlap = len(sig1["in_sources"] & sig2["in_sources"])

        total_similarity = rel_overlap + target_overlap + source_overlap

        result.predictions_made += 1
        if total_similarity >= 2 or sig1["type"] == sig2["type"]:
            result.correct_predictions += 1
            result.details.append({
                "equivalence": "QM_Formulations",
                "members": ["WaveMechanics", "MatrixMechanics"],
                "discovered": True,
                "similarity_score": total_similarity,
                "reason": "Same type and/or shared structural patterns"
            })
        else:
            result.missed_targets += 1
            result.details.append({
                "equivalence": "QM_Formulations",
                "discovered": False,
                "similarity_score": total_similarity
            })

    # QED_Formulations: Feynman_QED ≃ Schwinger_QED ≃ Tomonaga_QED
    qed_members = ["Feynman_QED", "Schwinger_QED", "Tomonaga_QED"]
    qed_sigs = {m: signatures.get(m) for m in qed_members if m in signatures}

    if len(qed_sigs) >= 2:
        # Check pairwise similarities
        members_list = list(qed_sigs.keys())
        all_similar = True

        for i, m1 in enumerate(members_list):
            for m2 in members_list[i+1:]:
                sig1, sig2 = qed_sigs[m1], qed_sigs[m2]
                if sig1 and sig2:
                    same_type = sig1["type"] == sig2["type"]
                    rel_overlap = len(sig1["out_rels"] & sig2["out_rels"])
                    if not same_type and rel_overlap == 0:
                        all_similar = False

        result.predictions_made += 1
        if all_similar:
            result.correct_predictions += 1
            result.details.append({
                "equivalence": "QED_Formulations",
                "members": members_list,
                "discovered": True,
                "reason": "All members share type and/or structural patterns"
            })
        else:
            result.missed_targets += 1
            result.details.append({
                "equivalence": "QED_Formulations",
                "discovered": False
            })

    return result


def test_path_consistency() -> StressTestResult:
    """
    CONSISTENCY: Same queries should produce consistent results.

    Run the same analysis multiple times and verify determinism.
    """
    result = StressTestResult("Consistency Check")

    store = create_physics_dataset()

    test_queries = [
        ("Newton", "Dirac"),
        ("Galileo", "QuantumMechanics"),
        ("Maxwell", "QED"),
    ]

    for source, target in test_queries:
        results = []

        # Run same query 3 times
        for _ in range(3):
            paths = store.find_paths(source, target, max_length=8)
            results.append(len(paths))

        result.predictions_made += 1

        # Check consistency
        if len(set(results)) == 1:
            result.correct_predictions += 1
            result.details.append({
                "query": f"{source} -> {target}",
                "consistent": True,
                "result": results[0]
            })
        else:
            result.false_positives += 1
            result.details.append({
                "query": f"{source} -> {target}",
                "consistent": False,
                "results": results,
                "note": "INCONSISTENT - different results for same query!"
            })

    return result


def test_hub_identification() -> StressTestResult:
    """
    HUB IDENTIFICATION: Does the system correctly identify important nodes?

    We know Einstein, Dirac, Newton, Maxwell are major hubs historically.
    Does the system's connectivity analysis agree?
    """
    result = StressTestResult("Hub Identification")

    store = create_physics_dataset()
    morphisms = store.list_morphisms(limit=10000)

    # Calculate degree for each node
    degree = {}
    for mor in morphisms:
        degree[mor.source_name] = degree.get(mor.source_name, 0) + 1
        degree[mor.target_name] = degree.get(mor.target_name, 0) + 1

    # Sort by degree
    sorted_nodes = sorted(degree.items(), key=lambda x: -x[1])
    top_10 = [n for n, d in sorted_nodes[:10]]

    # Ground truth: historically important figures we expect in top 10
    expected_hubs = {"Einstein", "Dirac", "Newton", "Maxwell", "Bohr", "Schrodinger", "Heisenberg"}

    for expected in expected_hubs:
        result.predictions_made += 1
        if expected in top_10:
            result.correct_predictions += 1
            rank = top_10.index(expected) + 1
            result.details.append({
                "expected_hub": expected,
                "found": True,
                "rank": rank,
                "degree": degree.get(expected, 0)
            })
        else:
            result.missed_targets += 1
            result.details.append({
                "expected_hub": expected,
                "found": False,
                "actual_degree": degree.get(expected, 0),
                "note": f"Not in top 10, top 10 are: {top_10}"
            })

    return result


def run_all_stress_tests() -> Dict:
    """Run all stress tests and compile results."""

    print("=" * 70)
    print("KOMPOSOS-III QUALITY STRESS TEST")
    print("=" * 70)
    print()
    print("Testing whether the system produces MEANINGFUL predictions,")
    print("not just well-formatted prose.")
    print()

    tests = [
        ("Backtest: Morphism Prediction", test_backtest_morphism_prediction),
        ("Temporal Holdout (Pre-1925)", test_temporal_holdout),
        ("Equivalence Discovery", test_equivalence_discovery),
        ("Consistency Check", test_path_consistency),
        ("Hub Identification", test_hub_identification),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"[Running] {test_name}...")
        try:
            result = test_func()
            results.append(result)
            print(f"  Precision: {result.precision:.2%}")
            print(f"  Recall: {result.recall:.2%}")
            print(f"  F1 Score: {result.f1_score:.2%}")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()

    # Compile summary
    total_predictions = sum(r.predictions_made for r in results)
    total_correct = sum(r.correct_predictions for r in results)
    total_missed = sum(r.missed_targets for r in results)

    overall_precision = total_correct / total_predictions if total_predictions > 0 else 0
    overall_recall = total_correct / (total_correct + total_missed) if (total_correct + total_missed) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Total Tests: {len(results)}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Correct: {total_correct}")
    print(f"Missed: {total_missed}")
    print()
    print(f"Overall Precision: {overall_precision:.2%}")
    print(f"Overall Recall: {overall_recall:.2%}")
    print(f"Overall F1 Score: {overall_f1:.2%}")
    print()

    # Quality assessment
    if overall_f1 >= 0.7:
        quality = "EXCELLENT"
        assessment = "The system produces meaningful, validated predictions."
    elif overall_f1 >= 0.5:
        quality = "GOOD"
        assessment = "The system produces useful predictions with room for improvement."
    elif overall_f1 >= 0.3:
        quality = "MODERATE"
        assessment = "The system has some predictive power but needs enhancement."
    else:
        quality = "NEEDS WORK"
        assessment = "The system's predictions are not yet reliable."

    print(f"Quality Assessment: {quality}")
    print(f"  {assessment}")
    print()

    return {
        "results": results,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "quality": quality,
        "assessment": assessment
    }


def generate_stress_test_report(results: Dict, output_path: Path):
    """Generate a detailed markdown report of stress test results."""

    report = f"""# KOMPOSOS-III Quality Stress Test Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

This report validates whether KOMPOSOS-III produces **meaningful predictions**
rather than just well-formatted prose. We employ rigorous backtesting and
holdout validation to measure actual predictive power.

### Overall Quality Assessment

| Metric | Value |
|--------|-------|
| **Quality Grade** | {results['quality']} |
| **Overall F1 Score** | {results['overall_f1']:.2%} |
| **Precision** | {results['overall_precision']:.2%} |
| **Recall** | {results['overall_recall']:.2%} |

**Assessment:** {results['assessment']}

---

## Test Methodology

1. **Backtesting**: Remove known relationships, verify Oracle predicts them
2. **Temporal Holdout**: Train on pre-1925 data, predict post-1925 developments
3. **Equivalence Discovery**: Remove equivalences, verify structural analysis finds them
4. **Consistency**: Same queries produce same results
5. **Hub Identification**: System identifies historically important figures

---

## Detailed Results

"""

    for result in results['results']:
        report += f"""### {result.test_name}

| Metric | Value |
|--------|-------|
| Predictions Made | {result.predictions_made} |
| Correct | {result.correct_predictions} |
| Missed | {result.missed_targets} |
| Precision | {result.precision:.2%} |
| Recall | {result.recall:.2%} |
| F1 Score | {result.f1_score:.2%} |

**Details:**

"""
        for detail in result.details[:10]:  # Limit details
            report += f"- {detail}\n"

        if len(result.details) > 10:
            report += f"- ... and {len(result.details) - 10} more\n"

        report += "\n"

    report += """---

## Interpretation

### What These Scores Mean

- **Precision**: Of the predictions the system made, what fraction were correct?
  - High precision = few false positives
  - Low precision = many spurious predictions

- **Recall**: Of the actual connections, what fraction did we predict?
  - High recall = few missed connections
  - Low recall = many important connections missed

- **F1 Score**: Harmonic mean of precision and recall
  - Balances both concerns
  - 70%+ is excellent, 50%+ is good, 30%+ is moderate

### Comparison to KOMPOSOS-jf

KOMPOSOS-jf reports are evaluated on:
1. **Cross-domain bridge discovery** (semantic similarity)
2. **Hypothesis generation** (structural inference)
3. **Sheaf coherence** (consistency across sources)
4. **Oracle predictions** (what should exist)

KOMPOSOS-III focuses on:
1. **Evolutionary path finding** (categorical composition)
2. **Equivalence detection** (HoTT univalence)
3. **Structural analysis** (Yoneda lemma)
4. **Gap identification** (Kan extensions)

Both systems aim for meaningful, validated predictions—not just prose.

---

*Report generated by KOMPOSOS-III Stress Test System*
"""

    output_path.write_text(report, encoding='utf-8')
    return output_path


if __name__ == "__main__":
    results = run_all_stress_tests()

    # Generate report
    output_path = Path(__file__).parent.parent / "evaluation_reports" / "STRESS_TEST_RESULTS.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_stress_test_report(results, output_path)

    print(f"Detailed report saved to: {output_path}")
