"""
KOMPOSOS-III Data Layer Tests
==============================

Comprehensive tests for the data layer:
1. Store operations (CRUD for objects, morphisms, paths, equivalences)
2. Path finding (evolution tracking)
3. Embeddings (semantic similarity)
4. Data sources (parsing and loading)
5. Integration (full pipeline)
"""

import pytest
import sys
import tempfile
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.store import (
    StoredObject, StoredMorphism, StoredPath, EquivalenceClass, HigherMorphism,
    KomposOSStore, create_memory_store, create_store
)
from data.sources import (
    ParsedWork, ParsedConcept, ParsedEntity,
    OpenAlexLoader, CustomDataLoader, BibTeXLoader
)
from data.config import KomposOSConfig, init_corpus, verify_corpus


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def memory_store():
    """Create an in-memory store for testing."""
    return create_memory_store()


@pytest.fixture
def populated_store(memory_store):
    """Create a store with sample data."""
    store = memory_store

    # Add physicists
    objects = [
        StoredObject("Newton", "Physicist", {"era": "classical", "birth": 1643}),
        StoredObject("Hamilton", "Physicist", {"era": "classical", "birth": 1805}),
        StoredObject("Lagrange", "Physicist", {"era": "classical", "birth": 1736}),
        StoredObject("Schrödinger", "Physicist", {"era": "quantum", "birth": 1887}),
        StoredObject("Heisenberg", "Physicist", {"era": "quantum", "birth": 1901}),
        StoredObject("Dirac", "Physicist", {"era": "quantum", "birth": 1902}),
    ]
    store.bulk_add_objects(objects)

    # Add relationships
    morphisms = [
        StoredMorphism("influenced", "Newton", "Lagrange", {"year": 1788}, 0.9),
        StoredMorphism("influenced", "Newton", "Hamilton", {"year": 1833}, 0.95),
        StoredMorphism("influenced", "Lagrange", "Hamilton", {"year": 1833}, 0.9),
        StoredMorphism("wave_mechanics", "Hamilton", "Schrödinger", {"year": 1926}),
        StoredMorphism("matrix_mechanics", "Hamilton", "Heisenberg", {"year": 1925}),
        StoredMorphism("unified", "Schrödinger", "Dirac", {"year": 1928}),
        StoredMorphism("unified", "Heisenberg", "Dirac", {"year": 1928}),
    ]
    store.bulk_add_morphisms(morphisms)

    return store


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Store Tests
# =============================================================================

class TestStoredObjects:
    """Test StoredObject operations."""

    def test_create_object(self):
        """Test creating a basic object."""
        obj = StoredObject("test", "TestType", {"key": "value"})
        assert obj.name == "test"
        assert obj.type_name == "TestType"
        assert obj.metadata["key"] == "value"
        assert obj.created_at is not None

    def test_add_and_get_object(self, memory_store):
        """Test adding and retrieving an object."""
        obj = StoredObject("Newton", "Physicist", {"era": "classical"})
        memory_store.add_object(obj)

        retrieved = memory_store.get_object("Newton")
        assert retrieved is not None
        assert retrieved.name == "Newton"
        assert retrieved.type_name == "Physicist"
        assert retrieved.metadata["era"] == "classical"

    def test_get_nonexistent_object(self, memory_store):
        """Test getting an object that doesn't exist."""
        result = memory_store.get_object("nonexistent")
        assert result is None

    def test_update_object(self, memory_store):
        """Test updating an existing object."""
        obj1 = StoredObject("test", "Type1", {"version": 1})
        memory_store.add_object(obj1)

        obj2 = StoredObject("test", "Type2", {"version": 2})
        memory_store.add_object(obj2)

        retrieved = memory_store.get_object("test")
        assert retrieved.type_name == "Type2"
        assert retrieved.metadata["version"] == 2

    def test_delete_object(self, memory_store):
        """Test deleting an object."""
        obj = StoredObject("to_delete", "TestType")
        memory_store.add_object(obj)

        assert memory_store.get_object("to_delete") is not None
        memory_store.delete_object("to_delete")
        assert memory_store.get_object("to_delete") is None

    def test_list_objects(self, populated_store):
        """Test listing objects with pagination."""
        all_objects = populated_store.list_objects(limit=100)
        assert len(all_objects) == 6

        first_three = populated_store.list_objects(limit=3)
        assert len(first_three) == 3

    def test_get_objects_by_type(self, populated_store):
        """Test filtering objects by type."""
        physicists = populated_store.get_objects_by_type("Physicist")
        assert len(physicists) == 6

        unknown = populated_store.get_objects_by_type("Unknown")
        assert len(unknown) == 0

    def test_count_objects(self, populated_store):
        """Test counting objects."""
        count = populated_store.count_objects()
        assert count == 6


class TestStoredMorphisms:
    """Test StoredMorphism operations."""

    def test_create_morphism(self):
        """Test creating a basic morphism."""
        mor = StoredMorphism("influenced", "A", "B", {"year": 1900}, 0.9)
        assert mor.name == "influenced"
        assert mor.source_name == "A"
        assert mor.target_name == "B"
        assert mor.confidence == 0.9
        assert mor.id == "influenced:A->B"

    def test_add_and_get_morphism(self, memory_store):
        """Test adding and retrieving a morphism."""
        memory_store.add_object(StoredObject("A", "Test"))
        memory_store.add_object(StoredObject("B", "Test"))

        mor = StoredMorphism("connects", "A", "B")
        memory_store.add_morphism(mor)

        retrieved = memory_store.get_morphism("connects:A->B")
        assert retrieved is not None
        assert retrieved.name == "connects"
        assert retrieved.source_name == "A"
        assert retrieved.target_name == "B"

    def test_get_morphisms_from(self, populated_store):
        """Test getting morphisms from a source."""
        from_newton = populated_store.get_morphisms_from("Newton")
        assert len(from_newton) == 2  # Newton -> Hamilton, Newton -> Lagrange

    def test_get_morphisms_to(self, populated_store):
        """Test getting morphisms to a target."""
        to_dirac = populated_store.get_morphisms_to("Dirac")
        assert len(to_dirac) == 2  # Schrödinger -> Dirac, Heisenberg -> Dirac

    def test_get_morphisms_by_name(self, populated_store):
        """Test getting morphisms by relationship type."""
        influenced = populated_store.get_morphisms_by_name("influenced")
        assert len(influenced) == 3

        unified = populated_store.get_morphisms_by_name("unified")
        assert len(unified) == 2


class TestPathFinding:
    """Test path finding for evolution tracking."""

    def test_find_direct_path(self, populated_store):
        """Test finding a direct path (single morphism)."""
        paths = populated_store.find_paths("Newton", "Hamilton")
        assert len(paths) >= 1
        assert paths[0].length == 1

    def test_find_multi_step_path(self, populated_store):
        """Test finding a multi-step path."""
        paths = populated_store.find_paths("Newton", "Dirac")
        assert len(paths) >= 1

        # Should find paths like Newton -> Hamilton -> Schrödinger -> Dirac
        lengths = [p.length for p in paths]
        assert any(l >= 2 for l in lengths)

    def test_find_multiple_paths(self, populated_store):
        """Test finding multiple paths to the same target."""
        paths = populated_store.find_paths("Newton", "Dirac", max_length=5)

        # Should find paths through both wave and matrix mechanics
        assert len(paths) >= 2

    def test_no_path_exists(self, populated_store):
        """Test when no path exists."""
        # Add an isolated node
        populated_store.add_object(StoredObject("Isolated", "Test"))

        paths = populated_store.find_paths("Isolated", "Dirac")
        assert len(paths) == 0

    def test_path_max_length(self, populated_store):
        """Test path finding respects max_length."""
        paths_short = populated_store.find_paths("Newton", "Dirac", max_length=2)
        paths_long = populated_store.find_paths("Newton", "Dirac", max_length=5)

        # Longer max_length should find at least as many paths
        assert len(paths_long) >= len(paths_short)


class TestEquivalenceClasses:
    """Test HoTT equivalence class operations."""

    def test_create_equivalence(self, memory_store):
        """Test creating an equivalence class."""
        equiv = EquivalenceClass(
            "QM_formulations",
            ["wave_mechanics", "matrix_mechanics"],
            equivalence_type="mathematical",
            witness="von_neumann_1932"
        )
        memory_store.add_equivalence(equiv)

        retrieved = memory_store.get_equivalence("QM_formulations")
        assert retrieved is not None
        assert "wave_mechanics" in retrieved.member_names
        assert "matrix_mechanics" in retrieved.member_names
        assert retrieved.witness == "von_neumann_1932"

    def test_are_equivalent(self, memory_store):
        """Test checking equivalence between objects."""
        equiv = EquivalenceClass(
            "equiv_class",
            ["A", "B", "C"],
            witness="proof"
        )
        memory_store.add_equivalence(equiv)

        result_ab = memory_store.are_equivalent("A", "B")
        assert result_ab is not None

        result_ad = memory_store.are_equivalent("A", "D")
        assert result_ad is None

    def test_find_equivalences_containing(self, memory_store):
        """Test finding all equivalence classes containing a member."""
        memory_store.add_equivalence(EquivalenceClass("class1", ["A", "B"]))
        memory_store.add_equivalence(EquivalenceClass("class2", ["B", "C"]))
        memory_store.add_equivalence(EquivalenceClass("class3", ["D", "E"]))

        classes_with_b = memory_store.find_equivalences_containing("B")
        assert len(classes_with_b) == 2


class TestHigherMorphisms:
    """Test 2-cell (path between paths) operations."""

    def test_add_higher_morphism(self, memory_store):
        """Test adding a 2-cell."""
        # First add some paths
        path1 = StoredPath("path1", ["m1", "m2"], "A", "C")
        path2 = StoredPath("path2", ["m3"], "A", "C")
        memory_store.add_path(path1)
        memory_store.add_path(path2)

        # Add higher morphism between paths
        hmor = HigherMorphism(
            "equivalence",
            path1.id,
            path2.id,
            "homotopy",
            witness="proof"
        )
        memory_store.add_higher_morphism(hmor)

        # Retrieve
        results = memory_store.get_higher_morphisms_between(path1.id, path2.id)
        assert len(results) >= 1


class TestStoreStatistics:
    """Test store statistics and analysis."""

    def test_get_statistics(self, populated_store):
        """Test getting store statistics."""
        stats = populated_store.get_statistics()

        assert "objects" in stats
        assert "morphisms" in stats
        assert "paths" in stats
        assert "equivalences" in stats
        assert "object_types" in stats
        assert "morphism_types" in stats

        assert stats["objects"] == 6
        assert stats["morphisms"] == 7

    def test_export_to_networkx(self, populated_store):
        """Test exporting to NetworkX graph."""
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")

        G = populated_store.export_to_networkx()

        assert len(G.nodes) == 6
        assert len(G.edges) == 7
        assert G.has_edge("Newton", "Hamilton")


# =============================================================================
# Data Sources Tests
# =============================================================================

class TestParsedWork:
    """Test ParsedWork data class."""

    def test_create_parsed_work(self):
        """Test creating a parsed work."""
        work = ParsedWork(
            id="W123",
            title="Test Paper",
            abstract="This is a test.",
            year=2020,
            authors=["Alice", "Bob"],
            concepts=["physics", "quantum"],
            citations=["W100", "W101"]
        )
        assert work.id == "W123"
        assert work.year == 2020
        assert len(work.authors) == 2

    def test_to_object(self):
        """Test converting ParsedWork to StoredObject."""
        work = ParsedWork(
            id="W123",
            title="Test Paper",
            year=2020,
            provenance="test"
        )
        obj = work.to_object()

        assert obj.name == "W123"
        assert obj.type_name == "Work"
        assert obj.metadata["title"] == "Test Paper"
        assert obj.metadata["year"] == 2020


class TestCustomDataLoader:
    """Test CustomDataLoader for CSV/JSON files."""

    def test_load_objects_csv(self, temp_dir):
        """Test loading objects from CSV."""
        csv_path = temp_dir / "objects.csv"
        with open(csv_path, 'w') as f:
            f.write("name,type,era\n")
            f.write("Newton,Physicist,classical\n")
            f.write("Hamilton,Physicist,classical\n")

        loader = CustomDataLoader()
        objects = list(loader.load_objects_csv(csv_path))

        assert len(objects) == 2
        assert objects[0].name == "Newton"
        assert objects[0].metadata["era"] == "classical"

    def test_load_morphisms_csv(self, temp_dir):
        """Test loading morphisms from CSV."""
        csv_path = temp_dir / "morphisms.csv"
        with open(csv_path, 'w') as f:
            f.write("name,source,target,year,confidence\n")
            f.write("influenced,Newton,Hamilton,1833,0.95\n")

        loader = CustomDataLoader()
        morphisms = list(loader.load_morphisms_csv(csv_path))

        assert len(morphisms) == 1
        assert morphisms[0].name == "influenced"
        assert morphisms[0].confidence == 0.95

    def test_load_json(self, temp_dir):
        """Test loading from JSON file."""
        json_path = temp_dir / "data.json"
        data = {
            "objects": [
                {"name": "A", "type": "Test", "metadata": {"key": "value"}}
            ],
            "morphisms": [
                {"name": "connects", "source": "A", "target": "B"}
            ]
        }
        with open(json_path, 'w') as f:
            json.dump(data, f)

        loader = CustomDataLoader()
        objects, morphisms = loader.load_json(json_path)

        assert len(objects) == 1
        assert len(morphisms) == 1
        assert objects[0].metadata["key"] == "value"


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfiguration:
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KomposOSConfig()

        assert config.embeddings_model == "all-mpnet-base-v2"
        assert config.log_level == "INFO"
        assert config.parallel_workers == 4

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = KomposOSConfig(parallel_workers=8)
        d = config.to_dict()

        assert d["parallel_workers"] == 8
        assert isinstance(d["db_path"], str)

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        d = {"parallel_workers": 8, "log_level": "DEBUG"}
        config = KomposOSConfig.from_dict(d)

        assert config.parallel_workers == 8
        assert config.log_level == "DEBUG"

    def test_config_save_and_load(self, temp_dir):
        """Test saving and loading config."""
        config = KomposOSConfig(parallel_workers=16)
        config_path = temp_dir / "config.json"
        config.save(config_path)

        loaded = KomposOSConfig.load(config_path)
        assert loaded.parallel_workers == 16

    def test_init_corpus(self, temp_dir):
        """Test initializing corpus directory."""
        corpus_path = temp_dir / "corpus"
        init_corpus(corpus_path)

        assert (corpus_path / "openalex" / "works").exists()
        assert (corpus_path / "pdfs" / "papers").exists()
        assert (corpus_path / "custom").exists()
        assert (corpus_path / "README.md").exists()

    def test_verify_corpus(self, temp_dir):
        """Test verifying corpus structure."""
        corpus_path = temp_dir / "corpus"
        init_corpus(corpus_path)

        results = verify_corpus(corpus_path)

        assert results["openalex/works"] is True
        assert results["custom"] is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full data pipeline."""

    def test_full_pipeline(self, temp_dir):
        """Test the complete data loading pipeline."""
        # Create corpus structure
        corpus_path = temp_dir / "corpus"
        init_corpus(corpus_path)

        # Create sample data
        objects_csv = corpus_path / "custom" / "objects.csv"
        with open(objects_csv, 'w') as f:
            f.write("name,type,era\n")
            f.write("Newton,Physicist,classical\n")
            f.write("Hamilton,Physicist,classical\n")
            f.write("Schrödinger,Physicist,quantum\n")

        morphisms_csv = corpus_path / "custom" / "morphisms.csv"
        with open(morphisms_csv, 'w') as f:
            f.write("name,source,target,year\n")
            f.write("influenced,Newton,Hamilton,1833\n")
            f.write("wave_mechanics,Hamilton,Schrödinger,1926\n")

        # Load into store
        store = create_memory_store()
        loader = CustomDataLoader()

        for obj in loader.load_objects_csv(objects_csv):
            store.add_object(obj)

        for mor in loader.load_morphisms_csv(morphisms_csv):
            store.add_morphism(mor)

        # Verify
        assert store.count_objects() == 3
        assert store.count_morphisms() == 2

        # Find paths
        paths = store.find_paths("Newton", "Schrödinger")
        assert len(paths) >= 1

    def test_temporal_queries(self, populated_store):
        """Test temporal evolution queries."""
        # Get objects in time range (by created_at)
        # Note: In a real scenario, we'd filter by year in metadata
        all_objs = populated_store.list_objects()

        classical = [o for o in all_objs if o.metadata.get("era") == "classical"]
        quantum = [o for o in all_objs if o.metadata.get("era") == "quantum"]

        assert len(classical) == 3  # Newton, Hamilton, Lagrange
        assert len(quantum) == 3    # Schrödinger, Heisenberg, Dirac

    def test_equivalence_with_paths(self, populated_store):
        """Test combining equivalence classes with path finding."""
        # Add equivalence between wave and matrix mechanics
        equiv = EquivalenceClass(
            "QM_formulations",
            ["wave_mechanics:Hamilton->Schrödinger", "matrix_mechanics:Hamilton->Heisenberg"],
            equivalence_type="mathematical",
            witness="von_neumann_1932"
        )
        populated_store.add_equivalence(equiv)

        # Find paths from Newton to Dirac
        paths = populated_store.find_paths("Newton", "Dirac", max_length=5)

        # Both paths should be found
        assert len(paths) >= 2

        # Check equivalence
        result = populated_store.are_equivalent(
            "wave_mechanics:Hamilton->Schrödinger",
            "matrix_mechanics:Hamilton->Heisenberg"
        )
        assert result is not None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
