"""
Example 01: Basic Query Flow

This example demonstrates the core KOMPOSOS-III query flow:
1. Build a category of concepts
2. Add paths (equivalences) between them
3. Use Kan extensions to predict unknown relationships
4. Use game theory to find stable answers

This is a simplified version - the full system uses Opus for NL parsing.
"""

import sys
sys.path.insert(0, '.')

from categorical.category import Category, obj, mor
from categorical.kan_extensions import KanExtensionOracle
from hott.identity import Path, IdentityType, refl, trans
from cubical.paths import PathType
from cubical.kan_ops import comp, hfill
from game.open_games import OpenGame, OpenGameCategory
from game.nash import TwoPlayerGame


def main():
    print("=" * 60)
    print("KOMPOSOS-III: Basic Query Example")
    print("=" * 60)

    # =========================================================
    # LAYER A: Build Categorical Structure
    # =========================================================
    print("\n[Layer A: Categorical Structure]")

    physics = Category("Physics")

    # Add concepts as objects
    newton = physics.add_object(obj("Newton", era="classical"))
    hamilton = physics.add_object(obj("Hamilton", era="classical"))
    schrodinger = physics.add_object(obj("Schrödinger", era="quantum"))
    heisenberg = physics.add_object(obj("Heisenberg", era="quantum"))
    dirac = physics.add_object(obj("Dirac", era="quantum"))

    # Add transformations as morphisms
    physics.add_morphism(mor("reformulation", newton, hamilton, year=1833))
    physics.add_morphism(mor("wave_mechanics", hamilton, schrodinger, year=1926))
    physics.add_morphism(mor("matrix_mechanics", hamilton, heisenberg, year=1925))
    physics.add_morphism(mor("unification", schrodinger, dirac, year=1928))
    physics.add_morphism(mor("unification", heisenberg, dirac, year=1928))

    print(f"Category: {physics}")
    print(f"Objects: {list(physics.objects.keys())}")

    # Find paths
    paths = physics.find_paths(newton, dirac)
    print(f"Paths from Newton to Dirac: {len(paths)}")
    for i, path in enumerate(paths):
        path_str = " → ".join(m.name for m in path)
        print(f"  Path {i+1}: {path_str}")

    # =========================================================
    # LAYER B: HoTT Identity
    # =========================================================
    print("\n[Layer B: HoTT Identity Types]")

    # Create paths representing equivalences
    wave_matrix_equiv = IdentityType(
        "Mechanics",
        "wave_mechanics",
        "matrix_mechanics"
    )
    equivalence_path = Path(
        wave_matrix_equiv,
        witness="von_neumann_proof",
        provenance="Mathematical equivalence proven 1932",
        confidence=1.0
    )
    print(f"Equivalence: {equivalence_path}")
    print(f"  Wave mechanics = Matrix mechanics (by von Neumann)")

    # Reflexivity
    id_path = refl("quantum_mechanics", "Theory")
    print(f"Reflexivity: {id_path}")

    # =========================================================
    # LAYER C: Cubical Paths
    # =========================================================
    print("\n[Layer C: Cubical Path Composition]")

    # Create computational paths
    p1 = PathType("Theory", "Newton", "Hamilton", provenance="reformulation")
    p2 = PathType("Theory", "Hamilton", "Schrödinger", provenance="wave_mechanics")

    # Compose paths
    composed = comp(p1, p2)
    print(f"Composed path: {composed}")
    print(f"  {composed.left} ~> {composed.right}")

    # =========================================================
    # LAYER D: Game Theory
    # =========================================================
    print("\n[Layer D: Game-Theoretic Equilibrium]")

    # Model the encoder/decoder game
    game = TwoPlayerGame(
        name="TheoryValidation",
        actions1=["propose_correct", "propose_incorrect"],
        actions2=["accept", "reject"],
        payoffs1={
            ("propose_correct", "accept"): 1.0,
            ("propose_correct", "reject"): -0.5,
            ("propose_incorrect", "accept"): 0.3,
            ("propose_incorrect", "reject"): -0.5,
        },
        payoffs2={
            ("propose_correct", "accept"): 1.0,
            ("propose_correct", "reject"): -1.0,
            ("propose_incorrect", "accept"): -1.0,
            ("propose_incorrect", "reject"): 0.5,
        }
    )

    print(game.display_matrix())

    equilibria = game.find_pure_nash()
    print(f"\nNash Equilibria: {equilibria}")
    print("The stable answer: (propose_correct, accept)")
    print("  → Encoder learns to propose correct answers")
    print("  → Decoder learns to accept correct ones")

    # =========================================================
    # SYNTHESIS
    # =========================================================
    print("\n" + "=" * 60)
    print("SYNTHESIS: Query Result")
    print("=" * 60)

    print("""
Query: "How did quantum mechanics evolve from classical physics?"

Answer (via categorical path-finding):
  Newton → Hamilton (1833: reformulation)
  Hamilton → Schrödinger (1926: wave mechanics)
     OR
  Hamilton → Heisenberg (1925: matrix mechanics)

  Both paths unify at Dirac (1928).

Equivalence (via HoTT):
  Wave mechanics ≃ Matrix mechanics (von Neumann, 1932)
  By univalence: they ARE the same theory.

Stability (via Game Theory):
  Nash equilibrium: (correct_answer, accept)
  This answer is game-theoretically stable.
""")


if __name__ == "__main__":
    main()
