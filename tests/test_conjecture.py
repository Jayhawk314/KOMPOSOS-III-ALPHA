"""
Tests for conjecture.py

Strategy: mock KomposOSStore and EmbeddingsEngine using the exact attribute
contracts the real strategies read (source_name, target_name, name, confidence,
type_name, metadata).  No real DB or embedding model needed.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Lightweight mocks that match the real data contracts
# ---------------------------------------------------------------------------

@dataclass
class MockMorphism:
    source_name: str
    target_name: str
    name: str                   # relation type, e.g. "influenced"
    confidence: float = 0.8


@dataclass
class MockObject:
    name: str
    type_name: str
    metadata: Dict = field(default_factory=dict)


class MockEmbeddingsEngine:
    """
    Similarity is driven by a lookup table.  Any pair not in the table
    returns 0.0, so tests are fully explicit about what's similar.
    """

    is_available = True

    def __init__(self, similarities: Optional[Dict] = None):
        # similarities: {frozenset({a, b}): score}
        self._sims = similarities or {}

    def similarity(self, a: str, b: str) -> float:
        key = frozenset((a, b))
        return self._sims.get(key, 0.0)


class MockStore:
    def __init__(self, objects: List[MockObject], morphisms: List[MockMorphism]):
        self._objects = {obj.name: obj for obj in objects}
        self._morphisms = morphisms

    def list_objects(self, limit: int = 10000) -> List[MockObject]:
        return list(self._objects.values())[:limit]

    def list_morphisms(self, limit: int = 50000) -> List[MockMorphism]:
        return self._morphisms[:limit]

    def get_object(self, name: str) -> Optional[MockObject]:
        return self._objects.get(name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A simple 4-node graph used by most tests:
#
#   A --influenced--> B --influenced--> C
#   A --influenced--> D
#
# Missing edges that generators should find:
#   A -> C  (composition: A->B->C)
#   B -> D  (structural hole: A -> both B and D)
#   D -> B  (structural hole: A -> both B and D, reverse)

OBJECTS_SIMPLE = [
    MockObject("A", "Physicist", {"birth": 1850, "era": "classical"}),
    MockObject("B", "Physicist", {"birth": 1880, "era": "classical"}),
    MockObject("C", "Physicist", {"birth": 1910, "era": "modern"}),
    MockObject("D", "Physicist", {"birth": 1870, "era": "classical"}),
]

MORPHISMS_SIMPLE = [
    MockMorphism("A", "B", "influenced", 0.9),
    MockMorphism("A", "D", "influenced", 0.85),
    MockMorphism("B", "C", "influenced", 0.8),
]


@pytest.fixture
def simple_store():
    return MockStore(OBJECTS_SIMPLE, MORPHISMS_SIMPLE)


@pytest.fixture
def simple_embeddings():
    # A and D are very similar; B and C are moderately similar
    return MockEmbeddingsEngine({
        frozenset(("A", "D")): 0.82,
        frozenset(("B", "C")): 0.65,
        frozenset(("A", "B")): 0.55,  # below 0.6 threshold
    })


# ---------------------------------------------------------------------------
# Import under test (after mocks are defined so the module can resolve)
# ---------------------------------------------------------------------------

# We need the conjecture module on the path.  It imports from oracle.prediction
# which imports from data — we stub those at module level so pytest can collect.

# Stub oracle.prediction so conjecture.py can import Prediction
@dataclass
class _StubPrediction:
    source: str
    target: str
    predicted_relation: str
    prediction_type: object = None
    strategy_name: str = "stub"
    confidence: float = 0.5
    reasoning: str = ""
    evidence: Dict = field(default_factory=dict)
    validated: Optional[bool] = None
    validation_source: Optional[str] = None

    @property
    def key(self):
        return (self.source, self.target, self.predicted_relation)

    def with_adjusted_confidence(self, c):
        import copy
        clone = copy.copy(self)
        clone.confidence = c
        return clone


# Monkey-patch the prediction module path so conjecture.py resolves
import types

_prediction_mod = types.ModuleType("oracle.prediction")
_prediction_mod.Prediction = _StubPrediction
_prediction_mod.PredictionType = type("PredictionType", (), {"ENSEMBLE": "ensemble"})
_prediction_mod.PredictionBatch = None
_prediction_mod.ConfidenceLevel = None

_oracle_pkg = types.ModuleType("oracle")
_oracle_pkg.prediction = _prediction_mod

sys.modules.setdefault("oracle", _oracle_pkg)
sys.modules.setdefault("oracle.prediction", _prediction_mod)
sys.modules.setdefault("oracle.coherence", types.ModuleType("oracle.coherence"))
sys.modules.setdefault("oracle.optimizer", types.ModuleType("oracle.optimizer"))
sys.modules.setdefault("data", types.ModuleType("data"))

# NOW import the module under test
sys.path.insert(0, str(Path(__file__).parent))
from conjecture import (
    _GraphCache,
    CompositionCandidates,
    StructuralHoleCandidates,
    FiberCandidates,
    SemanticCandidates,
    TemporalCandidates,
    YonedaCandidates,
    ConjectureEngine,
    Conjecture,
    ConjectureResult,
)


# ---------------------------------------------------------------------------
# Helper: build a cache from a store
# ---------------------------------------------------------------------------

def _cache(store: MockStore) -> _GraphCache:
    return _GraphCache(store)


# ===========================================================================
# _GraphCache tests
# ===========================================================================

class TestGraphCache:
    def test_morphisms_loaded_once(self, simple_store):
        cache = _cache(simple_store)
        m1 = cache.morphisms
        m2 = cache.morphisms
        assert m1 is m2  # same object, not re-fetched

    def test_existing_set(self, simple_store):
        cache = _cache(simple_store)
        assert ("A", "B") in cache.existing
        assert ("A", "C") not in cache.existing

    def test_outgoing_index(self, simple_store):
        cache = _cache(simple_store)
        assert len(cache.outgoing["A"]) == 2  # A->B, A->D
        assert len(cache.outgoing["B"]) == 1  # B->C

    def test_incoming_index(self, simple_store):
        cache = _cache(simple_store)
        assert len(cache.incoming["B"]) == 1  # A->B
        assert len(cache.incoming["C"]) == 1  # B->C

    def test_object_map(self, simple_store):
        cache = _cache(simple_store)
        assert cache.object_map["A"].type_name == "Physicist"


# ===========================================================================
# CompositionCandidates
# ===========================================================================

class TestCompositionCandidates:
    def test_finds_transitive_gap(self, simple_store):
        cache = _cache(simple_store)
        gen = CompositionCandidates(cache)
        candidates = gen.generate()

        # A->B->C exists, A->C does not => should be surfaced
        assert ("A", "C") in candidates

    def test_does_not_emit_existing_edge(self, simple_store):
        cache = _cache(simple_store)
        gen = CompositionCandidates(cache)
        candidates = gen.generate()

        # A->B already exists
        assert ("A", "B") not in candidates

    def test_does_not_emit_self_loop(self):
        # A->B->A: should NOT emit (A, A)
        store = MockStore(
            [MockObject("A", "P"), MockObject("B", "P")],
            [MockMorphism("A", "B", "r"), MockMorphism("B", "A", "r")],
        )
        cache = _cache(store)
        gen = CompositionCandidates(cache)
        candidates = gen.generate()
        assert ("A", "A") not in candidates
        assert ("B", "B") not in candidates

    def test_empty_graph(self):
        store = MockStore([MockObject("X", "T")], [])
        cache = _cache(store)
        gen = CompositionCandidates(cache)
        assert gen.generate() == set()


# ===========================================================================
# StructuralHoleCandidates
# ===========================================================================

class TestStructuralHoleCandidates:
    def test_common_ancestor_opens_both_directions(self, simple_store):
        cache = _cache(simple_store)
        gen = StructuralHoleCandidates(cache)
        candidates = gen.generate()

        # A -> B and A -> D, so (B, D) and (D, B) should both appear
        assert ("B", "D") in candidates
        assert ("D", "B") in candidates

    def test_common_descendant(self):
        # B->X and D->X exist, B->D does not
        store = MockStore(
            [MockObject("B", "P"), MockObject("D", "P"), MockObject("X", "T")],
            [MockMorphism("B", "X", "r"), MockMorphism("D", "X", "r")],
        )
        cache = _cache(store)
        gen = StructuralHoleCandidates(cache)
        candidates = gen.generate()
        assert ("B", "D") in candidates

    def test_does_not_emit_existing(self, simple_store):
        cache = _cache(simple_store)
        gen = StructuralHoleCandidates(cache)
        candidates = gen.generate()

        assert ("A", "B") not in candidates
        assert ("A", "D") not in candidates

    def test_sibling_cap_respected(self):
        # Star graph: hub -> 50 leaves.  Cap is 25 so we shouldn't explode.
        leaves = [MockObject(f"L{i}", "P") for i in range(50)]
        hub = MockObject("Hub", "P")
        morphisms = [MockMorphism("Hub", f"L{i}", "r") for i in range(50)]
        store = MockStore([hub] + leaves, morphisms)
        cache = _cache(store)
        gen = StructuralHoleCandidates(cache)
        # Should complete without timeout/memory issues
        candidates = gen.generate()
        # At most C(25,2)*2 = 600 pairs from ancestor pattern
        assert len(candidates) <= 25 * 24  # 600


# ===========================================================================
# FiberCandidates
# ===========================================================================

class TestFiberCandidates:
    def test_same_fiber_emitted(self, simple_store):
        cache = _cache(simple_store)
        gen = FiberCandidates(cache)
        candidates = gen.generate()

        # A, B, D are all (Physicist, classical) — pairs among them that aren't
        # already edges should appear.  B->D is missing.
        assert ("B", "D") in candidates
        assert ("D", "B") in candidates

    def test_different_fiber_not_emitted(self, simple_store):
        cache = _cache(simple_store)
        gen = FiberCandidates(cache)
        candidates = gen.generate()

        # C is (Physicist, modern), different fiber from A/B/D
        # (A, C) might appear from other generators but NOT from FiberCandidates
        # unless A and C share a fiber — they don't.
        # A->C is not in existing so it *could* be emitted if same fiber.
        # Verify C is not paired with classical-era objects by this generator.
        classical_members = {"A", "B", "D"}
        for src, tgt in candidates:
            if src == "C":
                assert tgt not in classical_members
            if tgt == "C":
                assert src not in classical_members

    def test_fiber_size_cap(self):
        # 50 objects in the same fiber
        objs = [MockObject(f"O{i}", "Physicist", {"era": "x"}) for i in range(50)]
        store = MockStore(objs, [])
        cache = _cache(store)
        gen = FiberCandidates(cache)
        candidates = gen.generate()
        # Capped at 30 members -> at most C(30,2)*2 = 870 pairs
        assert len(candidates) <= 30 * 29


# ===========================================================================
# SemanticCandidates
# ===========================================================================

class TestSemanticCandidates:
    def test_high_similarity_surfaced(self, simple_store, simple_embeddings):
        cache = _cache(simple_store)
        gen = SemanticCandidates(cache, simple_embeddings, top_k=10)
        candidates = gen.generate()

        # A and D have similarity 0.82 (above 0.6).  A->D exists already,
        # but D->A does not.
        assert ("D", "A") in candidates

    def test_below_threshold_not_surfaced(self, simple_store, simple_embeddings):
        cache = _cache(simple_store)
        gen = SemanticCandidates(cache, simple_embeddings, top_k=10)
        candidates = gen.generate()

        # A-B similarity is 0.55, below 0.6 threshold.  A->B exists anyway,
        # but the point is B->A should NOT appear from semantic alone.
        # (It might appear from other generators, but not this one.)
        # B->A: B and A have sim 0.55 < 0.6, so not a semantic candidate.
        assert ("B", "A") not in candidates

    def test_no_embeddings_returns_empty(self, simple_store):
        cache = _cache(simple_store)
        gen = SemanticCandidates(cache, embeddings=None, top_k=10)
        assert gen.generate() == set()

    def test_uses_top_neighbours_fast_path(self, simple_store):
        """If the engine has top_neighbours(), it should be called."""
        class FastEngine:
            is_available = True
            called_with = []

            def top_neighbours(self, target, k=10):
                self.called_with.append(target)
                return []  # empty is fine, we just check it was called

            def similarity(self, a, b):
                return 0.0

        eng = FastEngine()
        cache = _cache(simple_store)
        gen = SemanticCandidates(cache, eng, top_k=5)
        gen.generate()
        # Should have been called once per object
        assert len(eng.called_with) == len(simple_store.list_objects())


# ===========================================================================
# TemporalCandidates
# ===========================================================================

class TestTemporalCandidates:
    def test_influence_direction(self, simple_store):
        """Older -> younger should be emitted as influence candidate."""
        cache = _cache(simple_store)
        gen = TemporalCandidates(cache)
        candidates = gen.generate()

        # A(1850) is older than C(1910), types are (Physicist, Physicist) which
        # is in _VALID_TYPE_PAIRS.  A->C doesn't exist.
        assert ("A", "C") in candidates
        # Reverse should NOT be emitted (C is not older than A)
        assert ("C", "A") not in candidates

    def test_contemporary_both_directions(self):
        """Birth diff <= 20 -> both directions."""
        objs = [
            MockObject("X", "Physicist", {"birth": 1900}),
            MockObject("Y", "Physicist", {"birth": 1910}),  # diff = 10
        ]
        store = MockStore(objs, [])  # no existing edges
        cache = _cache(store)
        gen = TemporalCandidates(cache)
        candidates = gen.generate()

        assert ("X", "Y") in candidates
        assert ("Y", "X") in candidates

    def test_incompatible_type_pair_not_emitted(self):
        """(Theory, Physicist) is NOT in _VALID_TYPE_PAIRS."""
        objs = [
            MockObject("T1", "Theory", {"birth": 1900}),
            MockObject("P1", "Physicist", {"birth": 1950}),
        ]
        store = MockStore(objs, [])
        cache = _cache(store)
        gen = TemporalCandidates(cache)
        candidates = gen.generate()

        # Theory -> Physicist is not a valid type pair
        assert ("T1", "P1") not in candidates

    def test_missing_birth_metadata_skipped(self):
        objs = [
            MockObject("A", "Physicist", {"birth": 1900}),
            MockObject("B", "Physicist", {}),  # no birth
        ]
        store = MockStore(objs, [])
        cache = _cache(store)
        gen = TemporalCandidates(cache)
        candidates = gen.generate()

        assert ("A", "B") not in candidates
        assert ("B", "A") not in candidates


# ===========================================================================
# YonedaCandidates
# ===========================================================================

class TestYonedaCandidates:
    def test_high_hom_overlap_surfaced(self):
        """
        A and B both point to X and Y with same relation types.
        Yoneda similarity should be 1.0 -> (A,B) and (B,A) emitted.
        """
        objs = [
            MockObject("A", "P"), MockObject("B", "P"),
            MockObject("X", "T"), MockObject("Y", "T"),
        ]
        morphisms = [
            MockMorphism("A", "X", "influenced"),
            MockMorphism("A", "Y", "influenced"),
            MockMorphism("B", "X", "influenced"),
            MockMorphism("B", "Y", "influenced"),
        ]
        store = MockStore(objs, morphisms)
        cache = _cache(store)
        gen = YonedaCandidates(cache)
        candidates = gen.generate()

        assert ("A", "B") in candidates
        assert ("B", "A") in candidates

    def test_low_overlap_not_surfaced(self):
        """A and B share one target but have completely different relation types
        and different other targets -> low similarity."""
        objs = [
            MockObject("A", "P"), MockObject("B", "P"),
            MockObject("X", "T"), MockObject("Y", "T"), MockObject("Z", "T"),
        ]
        morphisms = [
            MockMorphism("A", "X", "influenced"),
            MockMorphism("A", "Y", "created"),
            MockMorphism("B", "X", "collaborated"),
            MockMorphism("B", "Z", "extended"),
        ]
        store = MockStore(objs, morphisms)
        cache = _cache(store)
        gen = YonedaCandidates(cache)
        candidates = gen.generate()

        # type overlap: {} (no shared types) -> 0
        # target overlap: {X} / {X,Y,Z} = 0.33
        # yoneda_sim = (0 + 0.33)/2 = 0.167 < 0.3
        assert ("A", "B") not in candidates

    def test_existing_edge_not_emitted(self):
        objs = [
            MockObject("A", "P"), MockObject("B", "P"), MockObject("X", "T"),
        ]
        morphisms = [
            MockMorphism("A", "X", "influenced"),
            MockMorphism("B", "X", "influenced"),
            MockMorphism("A", "B", "influenced"),  # already exists
        ]
        store = MockStore(objs, morphisms)
        cache = _cache(store)
        gen = YonedaCandidates(cache)
        candidates = gen.generate()

        assert ("A", "B") not in candidates
        # B->A does not exist and similarity is high, so it should appear
        assert ("B", "A") in candidates


# ===========================================================================
# Integration: ConjectureEngine
# ===========================================================================

class _MockOraclePredict:
    """
    Minimal stand-in for CategoricalOracle used by ConjectureEngine.
    Returns a single prediction per pair with confidence = 0.7.
    """

    def __init__(self, store, embeddings, min_confidence=0.4):
        self.store = store
        self.embeddings = embeddings
        self.min_confidence = min_confidence

    def predict(self, source: str, target: str):
        pred = _StubPrediction(
            source=source,
            target=target,
            predicted_relation="influenced",
            confidence=0.7,
            strategy_name="mock",
        )

        # Return a duck-typed result with .predictions
        class _Result:
            predictions = [pred]
        return _Result()


class TestConjectureEngine:
    def test_returns_conjecture_result(self, simple_store, simple_embeddings):
        oracle = _MockOraclePredict(simple_store, simple_embeddings)
        engine = ConjectureEngine(oracle)
        result = engine.conjecture(top_k=100)

        assert isinstance(result, ConjectureResult)
        assert result.pairs_surfaced > 0
        assert len(result.conjectures) > 0
        assert result.computation_time_ms > 0

    def test_conjectures_sorted_by_confidence(self, simple_store, simple_embeddings):
        oracle = _MockOraclePredict(simple_store, simple_embeddings)
        engine = ConjectureEngine(oracle)
        result = engine.conjecture(top_k=100)

        confs = [c.top_confidence for c in result.conjectures]
        assert confs == sorted(confs, reverse=True)

    def test_top_k_respected(self, simple_store, simple_embeddings):
        oracle = _MockOraclePredict(simple_store, simple_embeddings)
        engine = ConjectureEngine(oracle)
        result = engine.conjecture(top_k=2)

        assert len(result.conjectures) <= 2

    def test_min_confidence_filters(self, simple_store, simple_embeddings):
        """Mock returns 0.7; setting min to 0.8 should filter everything."""
        oracle = _MockOraclePredict(simple_store, simple_embeddings)
        engine = ConjectureEngine(oracle)
        result = engine.conjecture(top_k=100, min_confidence=0.8)

        assert len(result.conjectures) == 0

    def test_generator_whitelist(self, simple_store, simple_embeddings):
        oracle = _MockOraclePredict(simple_store, simple_embeddings)
        engine = ConjectureEngine(oracle)
        result = engine.conjecture(top_k=100, generators=["composition"])

        # Only composition ran
        assert "composition" in result.candidate_breakdown
        # Other generators should have 0 or not appear
        for name in ["structural_hole", "fiber", "semantic", "temporal", "yoneda"]:
            assert result.candidate_breakdown.get(name, 0) == 0

    def test_candidate_sources_tracked(self, simple_store, simple_embeddings):
        oracle = _MockOraclePredict(simple_store, simple_embeddings)
        engine = ConjectureEngine(oracle)
        result = engine.conjecture(top_k=100)

        # Every conjecture should have at least one source
        for c in result.conjectures:
            assert len(c.candidate_sources) >= 1

    def test_a_to_c_conjecture_found(self, simple_store, simple_embeddings):
        """A->C is the canonical transitive-closure gap in our fixture."""
        oracle = _MockOraclePredict(simple_store, simple_embeddings)
        engine = ConjectureEngine(oracle)
        result = engine.conjecture(top_k=100)

        pairs = {(c.source, c.target) for c in result.conjectures}
        assert ("A", "C") in pairs

    def test_no_existing_edges_in_output(self, simple_store, simple_embeddings):
        oracle = _MockOraclePredict(simple_store, simple_embeddings)
        engine = ConjectureEngine(oracle)
        result = engine.conjecture(top_k=100)

        existing = {(m.source_name, m.target_name) for m in simple_store.list_morphisms()}
        for c in result.conjectures:
            assert (c.source, c.target) not in existing


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_store(self):
        store = MockStore([], [])
        emb = MockEmbeddingsEngine()
        oracle = _MockOraclePredict(store, emb)
        engine = ConjectureEngine(oracle)
        result = engine.conjecture()

        assert result.conjectures == []
        assert result.pairs_surfaced == 0

    def test_single_object_no_morphisms(self):
        store = MockStore([MockObject("Lone", "P", {"birth": 1900})], [])
        emb = MockEmbeddingsEngine()
        oracle = _MockOraclePredict(store, emb)
        engine = ConjectureEngine(oracle)
        result = engine.conjecture()

        assert result.conjectures == []

    def test_fully_connected_graph(self):
        """Every pair already has an edge -> nothing to conjecture."""
        objs = [MockObject("A", "P"), MockObject("B", "P")]
        morphisms = [
            MockMorphism("A", "B", "r"),
            MockMorphism("B", "A", "r"),
        ]
        store = MockStore(objs, morphisms)
        emb = MockEmbeddingsEngine({frozenset(("A", "B")): 0.9})
        oracle = _MockOraclePredict(store, emb)
        engine = ConjectureEngine(oracle)
        result = engine.conjecture()

        assert result.conjectures == []

    def test_conjecture_best_property(self):
        c = Conjecture(
            source="A", target="B",
            predictions=[
                _StubPrediction("A", "B", "influenced", confidence=0.8),
                _StubPrediction("A", "B", "collaborated", confidence=0.5),
            ],
            top_confidence=0.8,
            candidate_sources=["composition"],
        )
        assert c.best.confidence == 0.8
        assert c.best.predicted_relation == "influenced"

    def test_conjecture_best_empty_predictions(self):
        c = Conjecture(
            source="A", target="B",
            predictions=[],
            top_confidence=0.0,
            candidate_sources=[],
        )
        assert c.best is None
