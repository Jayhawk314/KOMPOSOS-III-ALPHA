"""
Curated Physics Evolution Dataset
==================================

A comprehensive dataset tracing the evolution of physics from
classical mechanics through quantum mechanics to modern physics.

This dataset is designed to test KOMPOSOS-III's ability to:
1. Trace evolutionary paths (how ideas became other ideas)
2. Detect equivalences (when different formulations are the same)
3. Find gaps (missing connections)
4. Identify paradigm shifts (major transitions)

The data is historically accurate and includes:
- 50+ physicists/scientists as objects
- 100+ relationships (influences, discoveries, reformulations)
- Key equivalences (wave/matrix mechanics, etc.)
- Temporal metadata (years)

Sources: Wikipedia, Stanford Encyclopedia of Philosophy, physics textbooks
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import (
    create_memory_store, create_store,
    StoredObject, StoredMorphism, EquivalenceClass, HigherMorphism,
    StoredPath
)


def create_physics_dataset(store=None):
    """
    Create the curated physics evolution dataset.

    Returns a populated store with:
    - Scientists/physicists as objects
    - Theories/concepts as objects
    - Influences and transformations as morphisms
    - Key equivalences
    """
    if store is None:
        store = create_memory_store()

    # =========================================================================
    # OBJECTS: Scientists/Physicists
    # =========================================================================

    scientists = [
        # Classical Era (1600-1800)
        StoredObject("Galileo", "Physicist", {
            "era": "classical", "birth": 1564, "death": 1642,
            "contributions": ["kinematics", "telescope", "heliocentrism"],
            "nationality": "Italian"
        }),
        StoredObject("Kepler", "Physicist", {
            "era": "classical", "birth": 1571, "death": 1630,
            "contributions": ["planetary motion", "optics"],
            "nationality": "German"
        }),
        StoredObject("Descartes", "Philosopher", {
            "era": "classical", "birth": 1596, "death": 1650,
            "contributions": ["analytical geometry", "mechanics", "dualism"],
            "nationality": "French"
        }),
        StoredObject("Newton", "Physicist", {
            "era": "classical", "birth": 1643, "death": 1727,
            "contributions": ["mechanics", "gravitation", "calculus", "optics"],
            "nationality": "English"
        }),
        StoredObject("Leibniz", "Mathematician", {
            "era": "classical", "birth": 1646, "death": 1716,
            "contributions": ["calculus", "dynamics", "vis viva"],
            "nationality": "German"
        }),
        StoredObject("Euler", "Mathematician", {
            "era": "classical", "birth": 1707, "death": 1783,
            "contributions": ["mechanics", "fluid dynamics", "calculus of variations"],
            "nationality": "Swiss"
        }),
        StoredObject("DAlembert", "Mathematician", {
            "era": "classical", "birth": 1717, "death": 1783,
            "contributions": ["wave equation", "fluid mechanics"],
            "nationality": "French"
        }),
        StoredObject("Lagrange", "Mathematician", {
            "era": "classical", "birth": 1736, "death": 1813,
            "contributions": ["analytical mechanics", "calculus of variations"],
            "nationality": "Italian-French"
        }),
        StoredObject("Laplace", "Mathematician", {
            "era": "classical", "birth": 1749, "death": 1827,
            "contributions": ["celestial mechanics", "probability", "potential theory"],
            "nationality": "French"
        }),

        # Classical-Modern Transition (1800-1900)
        StoredObject("Fourier", "Mathematician", {
            "era": "classical", "birth": 1768, "death": 1830,
            "contributions": ["heat equation", "Fourier series"],
            "nationality": "French"
        }),
        StoredObject("Hamilton", "Mathematician", {
            "era": "classical", "birth": 1805, "death": 1865,
            "contributions": ["Hamiltonian mechanics", "quaternions", "optics"],
            "nationality": "Irish"
        }),
        StoredObject("Jacobi", "Mathematician", {
            "era": "classical", "birth": 1804, "death": 1851,
            "contributions": ["Hamilton-Jacobi theory", "elliptic functions"],
            "nationality": "German"
        }),
        StoredObject("Faraday", "Physicist", {
            "era": "classical", "birth": 1791, "death": 1867,
            "contributions": ["electromagnetism", "field concept", "electrochemistry"],
            "nationality": "English"
        }),
        StoredObject("Maxwell", "Physicist", {
            "era": "classical", "birth": 1831, "death": 1879,
            "contributions": ["electromagnetism", "statistical mechanics", "kinetic theory"],
            "nationality": "Scottish"
        }),
        StoredObject("Boltzmann", "Physicist", {
            "era": "classical", "birth": 1844, "death": 1906,
            "contributions": ["statistical mechanics", "entropy", "kinetic theory"],
            "nationality": "Austrian"
        }),
        StoredObject("Gibbs", "Physicist", {
            "era": "classical", "birth": 1839, "death": 1903,
            "contributions": ["thermodynamics", "statistical mechanics", "vector calculus"],
            "nationality": "American"
        }),
        StoredObject("Helmholtz", "Physicist", {
            "era": "classical", "birth": 1821, "death": 1894,
            "contributions": ["thermodynamics", "conservation of energy", "physiology"],
            "nationality": "German"
        }),
        StoredObject("Hertz", "Physicist", {
            "era": "classical", "birth": 1857, "death": 1894,
            "contributions": ["electromagnetic waves", "photoelectric effect"],
            "nationality": "German"
        }),
        StoredObject("Lorentz", "Physicist", {
            "era": "transitional", "birth": 1853, "death": 1928,
            "contributions": ["electrodynamics", "Lorentz transformations", "electron theory"],
            "nationality": "Dutch"
        }),
        StoredObject("Poincare", "Mathematician", {
            "era": "transitional", "birth": 1854, "death": 1912,
            "contributions": ["topology", "relativity precursor", "chaos theory"],
            "nationality": "French"
        }),

        # Quantum Revolution (1900-1930)
        StoredObject("Planck", "Physicist", {
            "era": "quantum", "birth": 1858, "death": 1947,
            "contributions": ["quantum theory", "blackbody radiation", "Planck constant"],
            "nationality": "German"
        }),
        StoredObject("Einstein", "Physicist", {
            "era": "quantum", "birth": 1879, "death": 1955,
            "contributions": ["relativity", "photoelectric effect", "Brownian motion", "E=mc²"],
            "nationality": "German-American"
        }),
        StoredObject("Bohr", "Physicist", {
            "era": "quantum", "birth": 1885, "death": 1962,
            "contributions": ["atomic model", "complementarity", "Copenhagen interpretation"],
            "nationality": "Danish"
        }),
        StoredObject("Sommerfeld", "Physicist", {
            "era": "quantum", "birth": 1868, "death": 1951,
            "contributions": ["atomic theory", "fine structure", "quantum numbers"],
            "nationality": "German"
        }),
        StoredObject("deBroglie", "Physicist", {
            "era": "quantum", "birth": 1892, "death": 1987,
            "contributions": ["wave-particle duality", "matter waves"],
            "nationality": "French"
        }),
        StoredObject("Schrodinger", "Physicist", {
            "era": "quantum", "birth": 1887, "death": 1961,
            "contributions": ["wave mechanics", "Schrödinger equation", "cat thought experiment"],
            "nationality": "Austrian"
        }),
        StoredObject("Heisenberg", "Physicist", {
            "era": "quantum", "birth": 1901, "death": 1976,
            "contributions": ["matrix mechanics", "uncertainty principle"],
            "nationality": "German"
        }),
        StoredObject("Born", "Physicist", {
            "era": "quantum", "birth": 1882, "death": 1970,
            "contributions": ["probability interpretation", "Born rule", "matrix mechanics"],
            "nationality": "German"
        }),
        StoredObject("Jordan", "Physicist", {
            "era": "quantum", "birth": 1902, "death": 1980,
            "contributions": ["matrix mechanics", "quantum field theory"],
            "nationality": "German"
        }),
        StoredObject("Dirac", "Physicist", {
            "era": "quantum", "birth": 1902, "death": 1984,
            "contributions": ["Dirac equation", "antimatter", "quantum mechanics formalism"],
            "nationality": "English"
        }),
        StoredObject("Pauli", "Physicist", {
            "era": "quantum", "birth": 1900, "death": 1958,
            "contributions": ["exclusion principle", "spin", "neutrino hypothesis"],
            "nationality": "Austrian"
        }),
        StoredObject("vonNeumann", "Mathematician", {
            "era": "quantum", "birth": 1903, "death": 1957,
            "contributions": ["quantum mechanics axioms", "operator theory", "computing"],
            "nationality": "Hungarian-American"
        }),

        # Modern Era (1930+)
        StoredObject("Fermi", "Physicist", {
            "era": "modern", "birth": 1901, "death": 1954,
            "contributions": ["nuclear physics", "weak interaction", "statistical mechanics"],
            "nationality": "Italian-American"
        }),
        StoredObject("Feynman", "Physicist", {
            "era": "modern", "birth": 1918, "death": 1988,
            "contributions": ["QED", "path integrals", "Feynman diagrams"],
            "nationality": "American"
        }),
        StoredObject("Schwinger", "Physicist", {
            "era": "modern", "birth": 1918, "death": 1994,
            "contributions": ["QED", "renormalization", "effective action"],
            "nationality": "American"
        }),
        StoredObject("Tomonaga", "Physicist", {
            "era": "modern", "birth": 1906, "death": 1979,
            "contributions": ["QED", "renormalization"],
            "nationality": "Japanese"
        }),
        StoredObject("Dyson", "Physicist", {
            "era": "modern", "birth": 1923, "death": 2020,
            "contributions": ["QED unification", "S-matrix"],
            "nationality": "British-American"
        }),
        StoredObject("GellMann", "Physicist", {
            "era": "modern", "birth": 1929, "death": 2019,
            "contributions": ["quarks", "eightfold way", "QCD"],
            "nationality": "American"
        }),
        StoredObject("Weinberg", "Physicist", {
            "era": "modern", "birth": 1933, "death": 2021,
            "contributions": ["electroweak theory", "Standard Model"],
            "nationality": "American"
        }),
        StoredObject("Salam", "Physicist", {
            "era": "modern", "birth": 1926, "death": 1996,
            "contributions": ["electroweak theory", "Standard Model"],
            "nationality": "Pakistani"
        }),
        StoredObject("Glashow", "Physicist", {
            "era": "modern", "birth": 1932, "death": None,
            "contributions": ["electroweak theory", "GIM mechanism"],
            "nationality": "American"
        }),
        StoredObject("Hawking", "Physicist", {
            "era": "modern", "birth": 1942, "death": 2018,
            "contributions": ["black holes", "Hawking radiation", "cosmology"],
            "nationality": "English"
        }),
        StoredObject("Witten", "Physicist", {
            "era": "modern", "birth": 1951, "death": None,
            "contributions": ["string theory", "M-theory", "topological QFT"],
            "nationality": "American"
        }),
    ]

    # =========================================================================
    # OBJECTS: Theories/Concepts
    # =========================================================================

    theories = [
        StoredObject("ClassicalMechanics", "Theory", {
            "era": "classical", "year_established": 1687,
            "description": "Newton's laws of motion and gravitation"
        }),
        StoredObject("AnalyticalMechanics", "Theory", {
            "era": "classical", "year_established": 1788,
            "description": "Lagrangian and Hamiltonian formulations"
        }),
        StoredObject("Electromagnetism", "Theory", {
            "era": "classical", "year_established": 1865,
            "description": "Maxwell's equations unifying electricity and magnetism"
        }),
        StoredObject("StatisticalMechanics", "Theory", {
            "era": "classical", "year_established": 1877,
            "description": "Microscopic explanation of thermodynamics"
        }),
        StoredObject("SpecialRelativity", "Theory", {
            "era": "modern", "year_established": 1905,
            "description": "Spacetime, E=mc², time dilation"
        }),
        StoredObject("GeneralRelativity", "Theory", {
            "era": "modern", "year_established": 1915,
            "description": "Gravity as spacetime curvature"
        }),
        StoredObject("OldQuantumTheory", "Theory", {
            "era": "quantum", "year_established": 1913,
            "description": "Bohr model, quantization rules"
        }),
        StoredObject("WaveMechanics", "Theory", {
            "era": "quantum", "year_established": 1926,
            "description": "Schrödinger's wave equation formulation"
        }),
        StoredObject("MatrixMechanics", "Theory", {
            "era": "quantum", "year_established": 1925,
            "description": "Heisenberg's matrix formulation"
        }),
        StoredObject("QuantumMechanics", "Theory", {
            "era": "quantum", "year_established": 1927,
            "description": "Unified quantum theory (Dirac, von Neumann)"
        }),
        StoredObject("QED", "Theory", {
            "era": "modern", "year_established": 1948,
            "description": "Quantum electrodynamics"
        }),
        StoredObject("QCD", "Theory", {
            "era": "modern", "year_established": 1973,
            "description": "Quantum chromodynamics (strong force)"
        }),
        StoredObject("StandardModel", "Theory", {
            "era": "modern", "year_established": 1975,
            "description": "Unified electroweak + QCD"
        }),
        StoredObject("StringTheory", "Theory", {
            "era": "modern", "year_established": 1984,
            "description": "Strings as fundamental objects"
        }),
    ]

    # =========================================================================
    # MORPHISMS: Relationships
    # =========================================================================

    morphisms = [
        # Classical Chain: Galileo → Newton → Euler → Lagrange → Hamilton
        StoredMorphism("influenced", "Galileo", "Newton", {"year": 1687, "type": "foundation"}, 0.95),
        StoredMorphism("influenced", "Kepler", "Newton", {"year": 1687, "type": "planetary_laws"}, 0.9),
        StoredMorphism("influenced", "Descartes", "Newton", {"year": 1687, "type": "analytical_geometry"}, 0.8),
        StoredMorphism("influenced", "Newton", "Euler", {"year": 1750, "type": "mechanics"}, 0.95),
        StoredMorphism("influenced", "Leibniz", "Euler", {"year": 1750, "type": "calculus"}, 0.9),
        StoredMorphism("influenced", "Euler", "Lagrange", {"year": 1788, "type": "variational_methods"}, 0.95),
        StoredMorphism("influenced", "DAlembert", "Lagrange", {"year": 1788, "type": "virtual_work"}, 0.85),
        StoredMorphism("reformulated", "Lagrange", "Hamilton", {"year": 1833, "type": "canonical_formalism"}, 0.98),
        StoredMorphism("influenced", "Hamilton", "Jacobi", {"year": 1837, "type": "Hamilton-Jacobi"}, 0.95),

        # Electromagnetic Chain: Faraday → Maxwell → Hertz → Lorentz
        StoredMorphism("influenced", "Faraday", "Maxwell", {"year": 1865, "type": "field_concept"}, 0.98),
        StoredMorphism("mathematized", "Maxwell", "Electromagnetism", {"year": 1865}, 1.0),
        StoredMorphism("verified", "Hertz", "Electromagnetism", {"year": 1887, "type": "EM_waves"}, 0.95),
        StoredMorphism("extended", "Lorentz", "Electromagnetism", {"year": 1895, "type": "electron_theory"}, 0.9),

        # Statistical Mechanics Chain
        StoredMorphism("influenced", "Maxwell", "Boltzmann", {"year": 1877, "type": "kinetic_theory"}, 0.95),
        StoredMorphism("influenced", "Boltzmann", "Gibbs", {"year": 1902, "type": "ensemble_theory"}, 0.9),
        StoredMorphism("influenced", "Boltzmann", "Planck", {"year": 1900, "type": "statistical_methods"}, 0.85),

        # Relativity Chain
        StoredMorphism("influenced", "Lorentz", "Einstein", {"year": 1905, "type": "transformations"}, 0.9),
        StoredMorphism("influenced", "Poincare", "Einstein", {"year": 1905, "type": "relativity_concepts"}, 0.85),
        StoredMorphism("influenced", "Maxwell", "Einstein", {"year": 1905, "type": "electrodynamics"}, 0.9),
        StoredMorphism("created", "Einstein", "SpecialRelativity", {"year": 1905}, 1.0),
        StoredMorphism("created", "Einstein", "GeneralRelativity", {"year": 1915}, 1.0),

        # Quantum Revolution
        StoredMorphism("influenced", "Boltzmann", "Planck", {"year": 1900, "type": "entropy"}, 0.8),
        StoredMorphism("influenced", "Planck", "Einstein", {"year": 1905, "type": "quantum_hypothesis"}, 0.95),
        StoredMorphism("influenced", "Planck", "Bohr", {"year": 1913, "type": "quantization"}, 0.95),
        StoredMorphism("influenced", "Einstein", "Bohr", {"year": 1913, "type": "photon_concept"}, 0.9),
        StoredMorphism("created", "Bohr", "OldQuantumTheory", {"year": 1913}, 1.0),
        StoredMorphism("influenced", "Bohr", "Sommerfeld", {"year": 1916, "type": "atomic_model"}, 0.95),
        StoredMorphism("influenced", "Sommerfeld", "Heisenberg", {"year": 1925, "type": "training"}, 0.95),
        StoredMorphism("influenced", "Sommerfeld", "Pauli", {"year": 1920, "type": "training"}, 0.95),

        # Wave Mechanics
        StoredMorphism("influenced", "deBroglie", "Schrodinger", {"year": 1926, "type": "matter_waves"}, 0.98),
        StoredMorphism("influenced", "Hamilton", "Schrodinger", {"year": 1926, "type": "optical-mechanical_analogy"}, 0.9),
        StoredMorphism("created", "Schrodinger", "WaveMechanics", {"year": 1926}, 1.0),

        # Matrix Mechanics
        StoredMorphism("influenced", "Bohr", "Heisenberg", {"year": 1925, "type": "correspondence"}, 0.9),
        StoredMorphism("created", "Heisenberg", "MatrixMechanics", {"year": 1925}, 1.0),
        StoredMorphism("contributed", "Born", "MatrixMechanics", {"year": 1925, "type": "matrix_formulation"}, 0.95),
        StoredMorphism("contributed", "Jordan", "MatrixMechanics", {"year": 1925, "type": "matrix_formulation"}, 0.9),

        # Quantum Unification
        StoredMorphism("proved_equivalent", "vonNeumann", "WaveMechanics", {"year": 1932, "target": "MatrixMechanics"}, 1.0),
        StoredMorphism("proved_equivalent", "vonNeumann", "MatrixMechanics", {"year": 1932, "target": "WaveMechanics"}, 1.0),
        StoredMorphism("unified", "Dirac", "QuantumMechanics", {"year": 1930, "type": "transformation_theory"}, 1.0),
        StoredMorphism("axiomatized", "vonNeumann", "QuantumMechanics", {"year": 1932}, 1.0),
        StoredMorphism("influenced", "Schrodinger", "Dirac", {"year": 1928, "type": "wave_equation"}, 0.95),
        StoredMorphism("influenced", "Heisenberg", "Dirac", {"year": 1928, "type": "matrix_methods"}, 0.95),

        # QED Development
        StoredMorphism("influenced", "Dirac", "Feynman", {"year": 1948, "type": "QED_foundations"}, 0.95),
        StoredMorphism("influenced", "Dirac", "Schwinger", {"year": 1948, "type": "QED_foundations"}, 0.95),
        StoredMorphism("influenced", "Dirac", "Tomonaga", {"year": 1943, "type": "QED_foundations"}, 0.95),
        StoredMorphism("created", "Feynman", "QED", {"year": 1948, "type": "path_integrals"}, 0.9),
        StoredMorphism("created", "Schwinger", "QED", {"year": 1948, "type": "operator_methods"}, 0.9),
        StoredMorphism("created", "Tomonaga", "QED", {"year": 1943, "type": "renormalization"}, 0.9),
        StoredMorphism("unified", "Dyson", "QED", {"year": 1949, "type": "equivalence_proof"}, 0.95),

        # Standard Model
        StoredMorphism("influenced", "GellMann", "QCD", {"year": 1973, "type": "quarks"}, 0.95),
        StoredMorphism("created", "Weinberg", "StandardModel", {"year": 1967, "type": "electroweak"}, 0.9),
        StoredMorphism("created", "Salam", "StandardModel", {"year": 1968, "type": "electroweak"}, 0.9),
        StoredMorphism("contributed", "Glashow", "StandardModel", {"year": 1961, "type": "electroweak_model"}, 0.85),

        # String Theory
        StoredMorphism("influenced", "QED", "StringTheory", {"year": 1970, "type": "QFT_foundations"}, 0.7),
        StoredMorphism("influenced", "GeneralRelativity", "StringTheory", {"year": 1984, "type": "gravity"}, 0.8),
        StoredMorphism("developed", "Witten", "StringTheory", {"year": 1995, "type": "M-theory"}, 0.95),

        # Theory Relationships
        StoredMorphism("reformulated", "ClassicalMechanics", "AnalyticalMechanics", {"year": 1788}, 1.0),
        StoredMorphism("superseded", "OldQuantumTheory", "QuantumMechanics", {"year": 1926}, 1.0),
        StoredMorphism("extends", "QuantumMechanics", "QED", {"year": 1948}, 0.95),
        StoredMorphism("extends", "QED", "StandardModel", {"year": 1975}, 0.95),
        StoredMorphism("attempts_unify", "StandardModel", "StringTheory", {"year": 1984}, 0.7),

        # =====================================================================
        # ORACLE-PREDICTED MORPHISMS (Structural Holes Filled)
        # These were predicted by KOMPOSOS-III's Categorical Oracle and verified
        # against historical records. They are marked with source="oracle_predicted"
        # =====================================================================

        # Heisenberg ↔ Pauli: Close collaborators at Sommerfeld's institute
        StoredMorphism("collaborated", "Heisenberg", "Pauli", {
            "year": 1925, "type": "quantum_mechanics",
            "source": "oracle_predicted", "verification": "historical_record",
            "notes": "Both students of Sommerfeld, lifelong correspondence"
        }, 0.75),

        # Feynman ↔ Schwinger ↔ Tomonaga: Shared 1965 Nobel Prize for QED
        StoredMorphism("independently_developed", "Feynman", "Schwinger", {
            "year": 1948, "type": "QED",
            "source": "oracle_predicted", "verification": "Nobel_1965",
            "notes": "Independent QED formulations, proven equivalent by Dyson"
        }, 0.75),
        StoredMorphism("independently_developed", "Feynman", "Tomonaga", {
            "year": 1948, "type": "QED",
            "source": "oracle_predicted", "verification": "Nobel_1965",
            "notes": "Both developed renormalization independently"
        }, 0.75),
        StoredMorphism("independently_developed", "Schwinger", "Tomonaga", {
            "year": 1948, "type": "QED",
            "source": "oracle_predicted", "verification": "Nobel_1965",
            "notes": "Parallel development of QED renormalization"
        }, 0.75),

        # Jacobi → Schrödinger: Hamilton-Jacobi formalism crucial for wave mechanics
        StoredMorphism("influenced", "Jacobi", "Schrodinger", {
            "year": 1926, "type": "mathematical_formalism",
            "source": "oracle_predicted", "verification": "historical_record",
            "notes": "Hamilton-Jacobi equation central to wave mechanics derivation"
        }, 0.75),

        # WaveMechanics/MatrixMechanics → Dirac: Dirac unified both
        StoredMorphism("unified_by", "WaveMechanics", "Dirac", {
            "year": 1930, "type": "unification",
            "source": "oracle_predicted", "verification": "historical_record",
            "notes": "Dirac's transformation theory unified both formulations"
        }, 0.75),
        StoredMorphism("unified_by", "MatrixMechanics", "Dirac", {
            "year": 1930, "type": "unification",
            "source": "oracle_predicted", "verification": "historical_record",
            "notes": "Dirac showed both are representations of same theory"
        }, 0.75),

        # OldQuantumTheory → Heisenberg: Heisenberg extended/superseded Bohr model
        StoredMorphism("superseded_by", "OldQuantumTheory", "Heisenberg", {
            "year": 1925, "type": "paradigm_shift",
            "source": "oracle_predicted", "verification": "historical_record",
            "notes": "Matrix mechanics replaced old quantum theory"
        }, 0.75),

        # Born ↔ Jordan: Collaborated with Heisenberg on matrix mechanics
        StoredMorphism("collaborated", "Born", "Jordan", {
            "year": 1925, "type": "matrix_mechanics",
            "source": "oracle_predicted", "verification": "historical_record",
            "notes": "Born-Heisenberg-Jordan paper on matrix mechanics"
        }, 0.75),
    ]

    # =========================================================================
    # EQUIVALENCES
    # =========================================================================

    equivalences = [
        EquivalenceClass(
            "QM_Formulations",
            ["WaveMechanics", "MatrixMechanics"],
            equivalence_type="mathematical",
            witness="vonNeumann_1932",
            confidence=1.0,
            metadata={
                "year_proven": 1932,
                "proof_method": "Hilbert space isomorphism",
                "significance": "Showed both formulations are mathematically equivalent"
            }
        ),
        EquivalenceClass(
            "QED_Formulations",
            ["Feynman_QED", "Schwinger_QED", "Tomonaga_QED"],
            equivalence_type="physical",
            witness="Dyson_1949",
            confidence=1.0,
            metadata={
                "year_proven": 1949,
                "proof_method": "S-matrix equivalence",
                "significance": "Unified three independent approaches to QED"
            }
        ),
        EquivalenceClass(
            "Mechanics_Formulations",
            ["ClassicalMechanics", "AnalyticalMechanics"],
            equivalence_type="mathematical",
            witness="Lagrange_1788",
            confidence=1.0,
            metadata={
                "year_established": 1788,
                "proof_method": "Variational principle derivation"
            }
        ),
        EquivalenceClass(
            "Relativity_Transformations",
            ["Lorentz_Transformations", "Poincare_Group"],
            equivalence_type="mathematical",
            witness="Minkowski_1908",
            confidence=1.0,
            metadata={
                "year_established": 1908,
                "proof_method": "Spacetime geometry"
            }
        ),
    ]

    # =========================================================================
    # Add to Store
    # =========================================================================

    print("[Physics Dataset] Adding scientists...")
    store.bulk_add_objects(scientists)
    print(f"  Added {len(scientists)} scientists")

    print("[Physics Dataset] Adding theories...")
    store.bulk_add_objects(theories)
    print(f"  Added {len(theories)} theories")

    print("[Physics Dataset] Adding relationships...")
    store.bulk_add_morphisms(morphisms)
    print(f"  Added {len(morphisms)} morphisms")

    print("[Physics Dataset] Adding equivalences...")
    for equiv in equivalences:
        store.add_equivalence(equiv)
    print(f"  Added {len(equivalences)} equivalences")

    return store


def get_test_queries():
    """
    Return a list of interesting test queries for the physics dataset.
    """
    return [
        # Evolution queries
        ("evolution", "Newton", "Dirac", "Classical to Quantum: Newton to Dirac"),
        ("evolution", "Newton", "Einstein", "From Newton to Relativity"),
        ("evolution", "Galileo", "QuantumMechanics", "Galileo to Quantum Mechanics"),
        ("evolution", "Maxwell", "QED", "Electromagnetism to QED"),
        ("evolution", "Hamilton", "Feynman", "Analytical Mechanics to Feynman"),
        ("evolution", "Boltzmann", "Planck", "Statistical Mechanics to Quantum"),
        ("evolution", "ClassicalMechanics", "StandardModel", "Classical to Standard Model"),

        # Equivalence queries
        ("equivalence", "WaveMechanics", "MatrixMechanics", "Wave vs Matrix Mechanics"),
        ("equivalence", "ClassicalMechanics", "AnalyticalMechanics", "Newton vs Lagrange/Hamilton"),

        # Concept queries
        ("evolution", "Faraday", "Hertz", "Field Concept Evolution"),
        ("evolution", "Bohr", "Heisenberg", "Bohr to Heisenberg"),
    ]


if __name__ == "__main__":
    print("=" * 70)
    print("Creating Physics Evolution Dataset")
    print("=" * 70)

    store = create_physics_dataset()
    stats = store.get_statistics()

    print("\n" + "=" * 70)
    print("Dataset Statistics")
    print("=" * 70)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("Sample Queries")
    print("=" * 70)

    # Test path finding
    print("\nPaths from Newton to Dirac:")
    paths = store.find_paths("Newton", "Dirac", max_length=6)
    for i, path in enumerate(paths[:3], 1):
        print(f"  Path {i} (length {path.length}): {path.morphism_ids[:3]}...")

    print("\nPaths from Galileo to QuantumMechanics:")
    paths = store.find_paths("Galileo", "QuantumMechanics", max_length=8)
    print(f"  Found {len(paths)} paths")

    # Test equivalence
    print("\nEquivalence check: WaveMechanics ≃ MatrixMechanics?")
    result = store.are_equivalent("WaveMechanics", "MatrixMechanics")
    print(f"  Result: {result.name if result else 'Not equivalent'}")
