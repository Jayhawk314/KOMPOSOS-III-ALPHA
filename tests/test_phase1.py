"""
Phase 1 Tests: Foundation (Categorical + Basic HoTT)

These tests verify the core functionality needed for Phase 1:
1. Basic category operations
2. Kan extensions
3. Identity types and paths
4. Path induction
"""

import pytest
import sys
sys.path.insert(0, '.')

from categorical.category import Category, Object, Morphism, obj, mor
from categorical.kan_extensions import (
    Functor, LeftKanExtension, RightKanExtension, KanExtensionOracle
)
from hott.identity import Path, IdentityType, refl, sym, trans, ap
from hott.path_induction import J, based_path_induction, transport


class TestCategory:
    """Test basic category operations."""

    def test_create_category(self):
        C = Category("Test")
        assert C.name == "Test"
        assert len(C.objects) == 0

    def test_add_object(self):
        C = Category("Test")
        a = C.add_object(obj("A"))
        assert "A" in C.objects
        assert C.objects["A"] == a

    def test_identity_morphism(self):
        C = Category("Test")
        a = C.add_object(obj("A"))
        id_a = C.identity(a)
        assert id_a.source == a
        assert id_a.target == a
        assert id_a.data.get("is_identity") == True

    def test_add_morphism(self):
        C = Category("Test")
        a = C.add_object(obj("A"))
        b = C.add_object(obj("B"))
        f = C.add_morphism(mor("f", a, b))
        assert "f" in C.morphisms
        assert f.source == a
        assert f.target == b

    def test_compose_morphisms(self):
        C = Category("Test")
        a = C.add_object(obj("A"))
        b = C.add_object(obj("B"))
        c = C.add_object(obj("C"))
        f = C.add_morphism(mor("f", a, b))
        g = C.add_morphism(mor("g", b, c))
        gf = C.compose(f, g)
        assert gf.source == a
        assert gf.target == c

    def test_compose_with_identity(self):
        C = Category("Test")
        a = C.add_object(obj("A"))
        b = C.add_object(obj("B"))
        f = C.add_morphism(mor("f", a, b))
        id_a = C.identity(a)
        # f ∘ id_a should equal f
        result = C.compose(id_a, f)
        assert result.source == f.source
        assert result.target == f.target

    def test_find_paths(self):
        C = Category("Test")
        a = C.add_object(obj("A"))
        b = C.add_object(obj("B"))
        c = C.add_object(obj("C"))
        f = C.add_morphism(mor("f", a, b))
        g = C.add_morphism(mor("g", b, c))
        paths = C.find_paths(a, c)
        assert len(paths) == 1
        assert len(paths[0]) == 2


class TestKanExtensions:
    """Test Kan extension computations."""

    def setup_method(self):
        """Set up test categories."""
        self.known = Category("Known")
        self.full = Category("Full")

        # Known objects
        self.e = self.known.add_object(obj("electron"))
        self.p = self.known.add_object(obj("proton"))

        # Full category includes unknown
        self.full.add_object(obj("electron"))
        self.full.add_object(obj("proton"))
        self.n = self.full.add_object(obj("neutron"))

        # Add morphisms
        self.full.add_morphism(mor("similar", self.p, self.n, weight=0.9))

    def test_functor_creation(self):
        F = Functor("test", self.known, self.full)
        F.add_object_mapping(self.e, {"charge": -1})
        assert F(self.e) == {"charge": -1}

    def test_left_kan_extension(self):
        # Create embedding K: Known → Full
        K = Functor("embed", self.known, self.full)
        K.add_object_mapping(self.e, self.full.objects["electron"])
        K.add_object_mapping(self.p, self.full.objects["proton"])

        # Create value functor F
        F = Functor("values", self.known, None)
        F.add_object_mapping(self.e, -1)  # charge
        F.add_object_mapping(self.p, +1)

        lan = LeftKanExtension(F, K)
        # Note: extension computation depends on morphisms in category
        # This is a basic smoke test
        assert lan.F == F
        assert lan.K == K

    def test_oracle(self):
        oracle = KanExtensionOracle(self.known, self.full)
        oracle.set_known_value(self.e, {"charge": -1, "mass": 0.511})
        oracle.set_known_value(self.p, {"charge": +1, "mass": 938.3})

        # Prediction for unknown should work (may return None if no paths)
        pred, conf = oracle.predict(self.n)
        # Even if no prediction, shouldn't crash
        assert conf >= 0.0


class TestIdentityTypes:
    """Test HoTT identity types."""

    def test_reflexivity(self):
        p = refl(42, "Int")
        assert p.source == 42
        assert p.target == 42
        assert p.witness == "refl"

    def test_symmetry(self):
        p = refl(42, "Int")
        q = sym(p)
        assert q.source == p.target
        assert q.target == p.source

    def test_transitivity(self):
        id1 = IdentityType("Num", 1, 2)
        id2 = IdentityType("Num", 2, 3)
        p = Path(id1, witness="step1")
        q = Path(id2, witness="step2")
        pq = trans(p, q)
        assert pq.source == 1
        assert pq.target == 3

    def test_action_on_paths(self):
        id_type = IdentityType("Int", 2, 2)
        p = refl(2, "Int")

        def double(x):
            return x * 2

        doubled = ap(double, p)
        assert doubled.source == 4
        assert doubled.target == 4


class TestPathInduction:
    """Test path induction (J eliminator)."""

    def test_j_on_refl(self):
        p = refl(5, "Int")

        def motive(a, b, path):
            return f"Property({a}, {b})"

        def base_case(a):
            return f"Base({a})"

        result = J("Int", motive, base_case, 5, 5, p)
        assert result == "Base(5)"

    def test_based_path_induction_on_refl(self):
        p = refl(5, "Int")

        def motive(b, path):
            return f"Prop({b})"

        result = based_path_induction("Int", 5, motive, "base_value", 5, p)
        assert result == "base_value"

    def test_transport_on_refl(self):
        p = refl(5, "Int")

        def P(n):
            return f"Fiber({n})"

        result = transport(P, p, "element")
        assert result == "element"


class TestIntegration:
    """Integration tests combining multiple layers."""

    def test_category_with_paths(self):
        """Category structure with HoTT path semantics."""
        C = Category("Physics")

        # Objects as concepts
        classical = C.add_object(obj("classical"))
        quantum = C.add_object(obj("quantum"))

        # Morphism as transformation
        f = C.add_morphism(mor("quantize", classical, quantum))

        # The morphism induces a path
        path_type = IdentityType("Theory", classical.name, quantum.name)
        quantization_path = Path(
            path_type,
            witness=f,
            provenance="physical_transformation"
        )

        assert quantization_path.source == "classical"
        assert quantization_path.target == "quantum"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
