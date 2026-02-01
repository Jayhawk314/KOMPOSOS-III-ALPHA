"""
Check which predictions are genuinely novel (not in STRING training data).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import csv
from data import KomposOSStore

def load_predictions(csv_path):
    """Load predictions from CSV."""
    predictions = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append({
                'rank': int(row['Rank']),
                'source': row['Source'],
                'target': row['Target'],
                'confidence': float(row['Confidence']),
                'system': row['System']
            })
    return predictions

def check_novelty(predictions, store):
    """Check which predictions are in training data."""
    # Get all known interactions
    all_morphisms = store.list_morphisms(limit=10000)
    known_edges = set()
    for m in all_morphisms:
        known_edges.add((m.source_name, m.target_name))
        known_edges.add((m.target_name, m.source_name))  # bidirectional

    print(f"Known edges in training data: {len(all_morphisms)}")
    print()

    results = []
    for pred in predictions:
        pair = (pred['source'], pred['target'])
        reverse_pair = (pred['target'], pred['source'])

        if pair in known_edges or reverse_pair in known_edges:
            novelty = "IN_TRAINING"
        else:
            novelty = "NOVEL"

        results.append({
            **pred,
            'novelty': novelty
        })

    return results

def analyze_novelty(results):
    """Compute novelty statistics."""
    total = len(results)
    novel = sum(1 for r in results if r['novelty'] == 'NOVEL')
    in_training = total - novel

    print("=" * 80)
    print("NOVELTY ANALYSIS")
    print("=" * 80)
    print()
    print(f"Total predictions: {total}")
    print(f"Novel (not in training): {novel} ({novel/total*100:.1f}%)")
    print(f"In training (confirmatory): {in_training} ({in_training/total*100:.1f}%)")
    print()

    # By system
    systems = set(r['system'] for r in results)
    for system in systems:
        system_results = [r for r in results if r['system'] == system]
        system_novel = sum(1 for r in system_results if r['novelty'] == 'NOVEL')
        print(f"{system} System:")
        print(f"  Total: {len(system_results)}")
        print(f"  Novel: {system_novel} ({system_novel/len(system_results)*100:.1f}%)")
        print()

def main():
    store = KomposOSStore('data/proteins/cancer_proteins.db')

    # Load predictions
    text_preds = load_predictions('reports/text_predictions_top50.csv')
    bio_preds = load_predictions('reports/bio_predictions_top50.csv')
    all_preds = text_preds + bio_preds

    # Check novelty
    results = check_novelty(all_preds, store)

    # Analyze
    analyze_novelty(results)

    # Export with novelty labels
    with open('reports/predictions_with_novelty.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rank', 'source', 'target', 'confidence', 'system', 'novelty'])
        writer.writeheader()
        writer.writerows(results)

    print("Saved results: reports/predictions_with_novelty.csv")

    # Summary by category
    print()
    print("=" * 80)
    print("NOVEL PREDICTIONS (Drug Target Candidates)")
    print("=" * 80)
    print()

    novel_results = [r for r in results if r['novelty'] == 'NOVEL']
    novel_results.sort(key=lambda x: x['confidence'], reverse=True)

    print("Top 20 Novel Predictions (Highest Confidence):")
    print()
    for i, r in enumerate(novel_results[:20], 1):
        print(f"{i:2d}. {r['source']:10s} -> {r['target']:10s} | conf={r['confidence']:.3f} | {r['system']}")

if __name__ == "__main__":
    main()
