"""
Comprehensive test suite for KOMPOSOS-III biological embeddings system.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import csv
from data import KomposOSStore
from data.bio_embeddings import BiologicalEmbeddingsEngine
from data.embeddings import EmbeddingsEngine

def test_file_exists(file_path: Path, description: str) -> bool:
    """Test if a file exists."""
    if file_path.exists():
        print(f"[PASS] {description}: {file_path}")
        return True
    else:
        print(f"[FAIL] {description}: {file_path} NOT FOUND")
        return False

def test_data_integrity():
    """Test data files integrity."""
    print("=" * 80)
    print("TEST 1: DATA FILE INTEGRITY")
    print("=" * 80)
    print()

    passed = 0
    total = 0

    # Check database
    total += 1
    if test_file_exists(Path('data/proteins/cancer_proteins.db'), "Cancer proteins database"):
        passed += 1
        # Check it can be opened
        try:
            store = KomposOSStore('data/proteins/cancer_proteins.db')
            proteins = store.list_objects(limit=100)
            print(f"  -> Contains {len(proteins)} proteins")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] Cannot open database: {e}")
        total += 1

    # Check sequences
    total += 1
    if test_file_exists(Path('data/proteins/sequences/all_sequences.fasta'), "All protein sequences"):
        passed += 1
        # Count sequences
        try:
            with open('data/proteins/sequences/all_sequences.fasta', 'r') as f:
                count = sum(1 for line in f if line.startswith('>'))
            print(f"  -> Contains {count} sequences")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] Cannot read sequences: {e}")
        total += 1

    # Check structures
    total += 1
    structures_dir = Path('data/proteins/structures')
    if structures_dir.exists():
        pdb_files = list(structures_dir.glob('*.pdb'))
        print(f"[PASS] AlphaFold structures directory: {len(pdb_files)} PDB files")
        passed += 1
    else:
        print(f"[FAIL] AlphaFold structures directory NOT FOUND")

    print()
    print(f"Data Integrity: {passed}/{total} tests passed")
    print()
    return passed == total

def test_embeddings_engines():
    """Test both embedding engines."""
    print("=" * 80)
    print("TEST 2: EMBEDDING ENGINES")
    print("=" * 80)
    print()

    passed = 0
    total = 0

    # Test biological embeddings
    total += 1
    try:
        bio_engine = BiologicalEmbeddingsEngine(device='cpu')
        print(f"[PASS] Biological embeddings engine initialized")
        print(f"  -> Model: {bio_engine.model_name}")
        print(f"  -> Dimension: {bio_engine.dimension}d")
        passed += 1

        # Test embedding
        total += 1
        try:
            emb = bio_engine.embed('KRAS', use_cache=True)
            print(f"[PASS] KRAS embedding: shape {emb.shape}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] Cannot generate embedding: {e}")

        # Test similarity
        total += 1
        try:
            sim = bio_engine.similarity('KRAS', 'NRAS')
            print(f"[PASS] KRAS vs NRAS similarity: {sim:.3f}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] Cannot compute similarity: {e}")

    except Exception as e:
        print(f"[FAIL] Cannot initialize biological embeddings: {e}")

    # Test text embeddings
    total += 1
    try:
        text_engine = EmbeddingsEngine()
        print(f"[PASS] Text embeddings engine initialized")
        print(f"  -> Dimension: {text_engine.dimension}d")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Cannot initialize text embeddings: {e}")

    print()
    print(f"Embedding Engines: {passed}/{total} tests passed")
    print()
    return passed == total

def test_predictions():
    """Test prediction outputs."""
    print("=" * 80)
    print("TEST 3: PREDICTION OUTPUTS")
    print("=" * 80)
    print()

    passed = 0
    total = 0

    # Check comparison JSON
    total += 1
    comp_file = Path('reports/bio_embeddings_comparison.json')
    if comp_file.exists():
        try:
            with open(comp_file, 'r') as f:
                data = json.load(f)

            print(f"[PASS] Comparison JSON: {comp_file}")
            print(f"  -> Biological precision: {data['bio_precision']*100:.1f}%")
            print(f"  -> Text precision: {data['text_precision']*100:.1f}%")
            print(f"  -> Bio predictions: {data['bio_total']}")
            print(f"  -> Text predictions: {data['text_total']}")

            # Check all predictions included
            if 'bio_all_predictions' in data and 'text_all_predictions' in data:
                print(f"  -> All predictions included: bio={len(data['bio_all_predictions'])}, text={len(data['text_all_predictions'])}")
                passed += 1
            else:
                print(f"  [FAIL] Missing full prediction lists")
        except Exception as e:
            print(f"[FAIL] Cannot read comparison JSON: {e}")
    else:
        print(f"[FAIL] Comparison JSON NOT FOUND: {comp_file}")

    # Check novelty results
    total += 1
    novelty_file = Path('reports/predictions_with_novelty.csv')
    if novelty_file.exists():
        try:
            with open(novelty_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            novel = sum(1 for r in rows if r['novelty'] == 'NOVEL')
            print(f"[PASS] Novelty analysis: {novelty_file}")
            print(f"  -> Total predictions: {len(rows)}")
            print(f"  -> Novel: {novel} ({novel/len(rows)*100:.1f}%)")
            passed += 1
        except Exception as e:
            print(f"[FAIL] Cannot read novelty CSV: {e}")
    else:
        print(f"[FAIL] Novelty CSV NOT FOUND: {novelty_file}")

    # Check therapeutic opportunities
    total += 1
    therapeutic_file = Path('reports/therapeutic_opportunities.csv')
    if therapeutic_file.exists():
        try:
            with open(therapeutic_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            tier1 = sum(1 for r in rows if r['tier'] == '1')
            tier2 = sum(1 for r in rows if r['tier'] == '2')
            tier3 = sum(1 for r in rows if r['tier'] == '3')

            print(f"[PASS] Therapeutic opportunities: {therapeutic_file}")
            print(f"  -> Total: {len(rows)}")
            print(f"  -> Tier 1 (both druggable): {tier1}")
            print(f"  -> Tier 2 (one druggable): {tier2}")
            print(f"  -> Tier 3 (research): {tier3}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] Cannot read therapeutic CSV: {e}")
    else:
        print(f"[FAIL] Therapeutic CSV NOT FOUND: {therapeutic_file}")

    # Check summary
    total += 1
    summary_file = Path('reports/GOOGLE_DEEPMIND_SUMMARY.md')
    if test_file_exists(summary_file, "Google/DeepMind summary"):
        passed += 1

    print()
    print(f"Prediction Outputs: {passed}/{total} tests passed")
    print()
    return passed == total

def test_top_predictions():
    """Verify top predictions are reasonable."""
    print("=" * 80)
    print("TEST 4: TOP PREDICTION VALIDATION")
    print("=" * 80)
    print()

    passed = 0
    total = 0

    # Load therapeutic opportunities
    therapeutic_file = Path('reports/therapeutic_opportunities.csv')
    if not therapeutic_file.exists():
        print("[FAIL] Cannot test - therapeutic file missing")
        return False

    with open(therapeutic_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check Tier 1 has reasonable number
    tier1 = [r for r in rows if r['tier'] == '1']
    total += 1
    if len(tier1) >= 10:
        print(f"[PASS] Tier 1 opportunities: {len(tier1)} (target: >=10)")
        passed += 1
    else:
        print(f"[FAIL] Tier 1 opportunities: {len(tier1)} (target: >=10)")

    # Check top prediction has high score
    total += 1
    if rows:
        top = rows[0]
        score = float(top['therapeutic_score'])
        if score > 0.5:
            print(f"[PASS] Top prediction score: {score:.4f} (target: >0.5)")
            print(f"  -> {top['source']} -> {top['target']}")
            print(f"  -> Drugs: {top['source_drugs']} / {top['target_drugs']}")
            passed += 1
        else:
            print(f"[FAIL] Top prediction score too low: {score:.4f}")

    # Check novelty rate
    novelty_file = Path('reports/predictions_with_novelty.csv')
    if novelty_file.exists():
        with open(novelty_file, 'r') as f:
            reader = csv.DictReader(f)
            nov_rows = list(reader)

        novel = sum(1 for r in nov_rows if r['novelty'] == 'NOVEL')
        novelty_rate = novel / len(nov_rows)

        total += 1
        if novelty_rate >= 0.80:
            print(f"[PASS] Novelty rate: {novelty_rate*100:.1f}% (target: >=80%)")
            passed += 1
        else:
            print(f"[FAIL] Novelty rate: {novelty_rate*100:.1f}% (target: >=80%)")

    print()
    print(f"Top Prediction Validation: {passed}/{total} tests passed")
    print()
    return passed == total

def main():
    """Run all tests."""
    print()
    print("=" * 80)
    print("KOMPOSOS-III COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()

    results = []

    # Run tests
    results.append(("Data Integrity", test_data_integrity()))
    results.append(("Embedding Engines", test_embeddings_engines()))
    results.append(("Prediction Outputs", test_predictions()))
    results.append(("Top Predictions", test_top_predictions()))

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "[PASS]" if passed else "[FAIL]"
        print(f"{symbol} {name}: {status}")

    print()

    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)

    if total_passed == total_tests:
        print(f"ALL TESTS PASSED ({total_passed}/{total_tests})")
        print()
        print("System ready for Google/DeepMind presentation!")
        return 0
    else:
        print(f"SOME TESTS FAILED ({total_passed}/{total_tests})")
        print()
        print("Fix issues before presenting.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
