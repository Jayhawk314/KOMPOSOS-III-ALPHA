"""
Analyze hub protein clustering in predictions.

Checks if predictions cluster on high-degree hub proteins, which could indicate:
1. Real biology (hubs are genuinely important)
2. Method artifact (embeddings for hubs similar to everything)

This analysis helps distinguish signal from noise.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
from collections import Counter
from typing import Dict, List, Tuple


def load_predictions(csv_path: str) -> List[Dict]:
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


def analyze_protein_frequency(predictions: List[Dict]) -> Counter:
    """Count how often each protein appears in predictions."""
    protein_counts = Counter()

    for pred in predictions:
        protein_counts[pred['source']] += 1
        protein_counts[pred['target']] += 1

    return protein_counts


def get_hub_proteins(protein_counts: Counter, threshold: int = 5) -> List[Tuple[str, int]]:
    """Identify hub proteins (appear in many predictions)."""
    hubs = [(protein, count) for protein, count in protein_counts.items() if count >= threshold]
    return sorted(hubs, key=lambda x: x[1], reverse=True)


def analyze_hub_predictions(predictions: List[Dict], hub_proteins: List[str]) -> Dict:
    """Analyze predictions involving hub proteins."""
    hub_set = set(hub_proteins)

    total = len(predictions)
    involving_hubs = sum(1 for p in predictions if p['source'] in hub_set or p['target'] in hub_set)
    both_hubs = sum(1 for p in predictions if p['source'] in hub_set and p['target'] in hub_set)

    return {
        'total': total,
        'involving_hubs': involving_hubs,
        'involving_hubs_pct': involving_hubs / total * 100,
        'both_hubs': both_hubs,
        'both_hubs_pct': both_hubs / total * 100
    }


def print_analysis(protein_counts: Counter, hub_analysis: Dict, system: str):
    """Print detailed hub clustering analysis."""
    print(f"\n{'='*80}")
    print(f"HUB CLUSTERING ANALYSIS - {system.upper()} SYSTEM")
    print(f"{'='*80}\n")

    # Top proteins
    print("Top 15 Most Frequent Proteins:")
    print(f"{'Rank':<6} {'Protein':<12} {'Count':<8} {'% of edges'}")
    print("-" * 50)

    total_edges = hub_analysis['total']
    for i, (protein, count) in enumerate(protein_counts.most_common(15), 1):
        pct = count / (total_edges * 2) * 100  # *2 because each edge has 2 proteins
        print(f"{i:<6} {protein:<12} {count:<8} {pct:.1f}%")

    print()

    # Hub statistics
    hubs = get_hub_proteins(protein_counts, threshold=5)
    print(f"Hub Proteins (appearing >= 5 times): {len(hubs)}")
    print()

    for protein, count in hubs[:10]:
        print(f"  {protein:<12} {count:>3} appearances")

    print()
    print("Hub Involvement Statistics:")
    print(f"  Total predictions: {hub_analysis['total']}")
    print(f"  Predictions involving >= 1 hub: {hub_analysis['involving_hubs']} ({hub_analysis['involving_hubs_pct']:.1f}%)")
    print(f"  Predictions with both hubs: {hub_analysis['both_hubs']} ({hub_analysis['both_hubs_pct']:.1f}%)")
    print()


def assess_hub_clustering(hub_analysis: Dict) -> str:
    """Provide assessment of hub clustering severity."""
    pct = hub_analysis['involving_hubs_pct']

    if pct < 40:
        return "LOW - Predictions distributed across many proteins"
    elif pct < 60:
        return "MODERATE - Some hub concentration, but diverse predictions"
    elif pct < 75:
        return "HIGH - Predictions strongly concentrated on hub proteins"
    else:
        return "SEVERE - Predictions dominated by hub proteins (likely artifact)"


def main():
    """Main analysis pipeline."""
    print()
    print("="*80)
    print("KOMPOSOS-III HUB CLUSTERING ANALYSIS")
    print("="*80)
    print()
    print("Purpose: Determine if predictions cluster on hub proteins")
    print("(which could indicate method artifact vs. real biology)")
    print()

    # Analyze biological predictions
    bio_preds = load_predictions('reports/bio_predictions_top50.csv')
    bio_counts = analyze_protein_frequency(bio_preds)
    bio_hubs = [p for p, c in get_hub_proteins(bio_counts, threshold=5)]
    bio_analysis = analyze_hub_predictions(bio_preds, bio_hubs)

    print_analysis(bio_counts, bio_analysis, "Biological (ESM-2)")

    print("="*80)
    print("ASSESSMENT")
    print("="*80)
    print()

    assessment = assess_hub_clustering(bio_analysis)
    print(f"Hub Clustering Severity: {assessment}")
    print()

    # Interpretation
    print("Interpretation:")
    print()

    if bio_analysis['involving_hubs_pct'] > 60:
        print("WARNING: High hub concentration detected")
        print()
        print("  The top 5 proteins appear in >60% of predictions.")
        print("  This suggests predictions may cluster on hub proteins.")
        print()
        print("  Possible explanations:")
        print("  1. Hub proteins ARE genuinely important (real biology)")
        print("  2. ESM-2 embeddings for hubs are similar to many proteins (artifact)")
        print("  3. Categorical strategies favor high-degree nodes (method bias)")
        print()
        print("  Next steps:")
        print("  - Experimental validation of hub-involving predictions")
        print("  - Compare to random baseline (shuffle protein names)")
        print("  - Test if hub clustering persists in larger datasets")
    else:
        print("OK: Hub concentration within expected range")
        print()
        print("  Predictions distributed across diverse proteins.")
        print("  No evidence of severe hub clustering artifact.")

    print()
    print("="*80)

    # Save report
    report_file = Path("reports/hub_clustering_analysis.txt")
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HUB CLUSTERING ANALYSIS\n")
        f.write("="*80 + "\n\n")

        f.write("Top 15 Proteins by Frequency:\n\n")
        for i, (protein, count) in enumerate(bio_counts.most_common(15), 1):
            pct = count / (bio_analysis['total'] * 2) * 100
            f.write(f"{i:2d}. {protein:<12} {count:>3} ({pct:>5.1f}%)\n")

        f.write(f"\n\nHub Involvement:\n")
        f.write(f"  Total predictions: {bio_analysis['total']}\n")
        f.write(f"  Involving hubs: {bio_analysis['involving_hubs']} ({bio_analysis['involving_hubs_pct']:.1f}%)\n")
        f.write(f"  Both hubs: {bio_analysis['both_hubs']} ({bio_analysis['both_hubs_pct']:.1f}%)\n")
        f.write(f"\nAssessment: {assessment}\n")

    print(f"\nReport saved: {report_file}")
    print()


if __name__ == "__main__":
    main()
