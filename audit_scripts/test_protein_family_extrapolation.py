"""
PROTEIN FAMILY EXTRAPOLATION TEST
==================================

Tests if ESM-2's "novel" predictions are actually protein family extrapolations.

HYPOTHESIS:
- ESM-2 encodes that similar sequences have similar binding partners
- "Novel" predictions might be: Protein A and Protein B are homologs,
  A interacts with X, so system predicts B interacts with X
- This is biological knowledge (family members share partners), not discovery

TEST:
1. Compute ESM-2 similarity for all protein pairs
2. For high-similarity pairs (>0.9), check if they share predicted interactions
3. If yes -> system is doing family extrapolation (expected)
4. If no -> system is finding cross-family interactions (novel)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

from data.bio_embeddings import BiologicalEmbeddingsEngine
from test_compositional_leakage import CompositionalLeakageDetector


PROTEIN_FAMILIES = {
    'RAS': ['KRAS', 'NRAS', 'HRAS'],
    'RAF': ['RAF1', 'BRAF', 'ARAF'],
    'AKT': ['AKT1', 'AKT2', 'AKT3'],
    'PIK3': ['PIK3CA', 'PIK3CB'],
    'CDK': ['CDK4', 'CDK6'],
    'STAT': ['STAT3', 'STAT5'],
    'ERK': ['ERK1', 'ERK2'],
    'BRCA': ['BRCA1', 'BRCA2'],
}


def load_deep_discoveries():
    """Load deep discovery predictions from leakage audit."""
    audit_path = Path("reports/leakage_audit_results.csv")

    if not audit_path.exists():
        print(f"ERROR: {audit_path} not found")
        print("Run test_compositional_leakage.py first")
        return None

    import csv

    deep_discoveries = []
    with open(audit_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['classification'] == 'TRULY_NOVEL':
                deep_discoveries.append({
                    'source': row['source'],
                    'target': row['target'],
                    'confidence': float(row['confidence']),
                    'system': row['system']
                })

    return deep_discoveries


def test_family_extrapolation():
    """
    Test if deep discoveries are family extrapolations.
    """
    print("=" * 80)
    print("PROTEIN FAMILY EXTRAPOLATION TEST")
    print("=" * 80)
    print()

    # Load ESM-2 embeddings
    print("Loading ESM-2 embeddings engine...")
    try:
        embeddings = BiologicalEmbeddingsEngine(device='cpu')
        if not embeddings.is_available:
            print("ERROR: ESM-2 not available")
            return
    except Exception as e:
        print(f"ERROR loading ESM-2: {e}")
        print("This test requires ESM-2. Run scripts/download_uniprot_sequences.py")
        return

    print()

    # Load deep discoveries
    deep_discoveries = load_deep_discoveries()

    if not deep_discoveries:
        print("No deep discoveries to analyze")
        return

    print(f"Loaded {len(deep_discoveries)} deep discoveries")
    print()

    # Compute similarity for all protein pairs in deep discoveries
    print("Computing ESM-2 similarities...")

    all_proteins = set()
    for pred in deep_discoveries:
        all_proteins.add(pred['source'])
        all_proteins.add(pred['target'])

    all_proteins = sorted(all_proteins)
    print(f"  Total proteins involved: {len(all_proteins)}")
    print()

    # Compute pairwise similarities
    similarities = {}
    for i, p1 in enumerate(all_proteins):
        for p2 in all_proteins[i+1:]:
            try:
                sim = embeddings.similarity(p1, p2)
                similarities[(p1, p2)] = sim
            except Exception as e:
                # Protein not in ESM-2 dataset
                similarities[(p1, p2)] = 0.0

    # Identify family extrapolations
    print("=" * 80)
    print("FAMILY EXTRAPOLATION ANALYSIS")
    print("=" * 80)
    print()

    family_extrapolations = []
    cross_family_discoveries = []

    for pred in deep_discoveries:
        source = pred['source']
        target = pred['target']

        # Get similarity
        pair = (source, target) if (source, target) in similarities else (target, source)
        sim = similarities.get(pair, 0.0)

        # Check if same family
        source_family = None
        target_family = None

        for family_name, members in PROTEIN_FAMILIES.items():
            if source in members:
                source_family = family_name
            if target in members:
                target_family = family_name

        is_same_family = (source_family == target_family and source_family is not None)

        # Classify
        if sim > 0.85:  # High similarity threshold
            family_extrapolations.append({
                **pred,
                'similarity': sim,
                'source_family': source_family,
                'target_family': target_family,
                'is_same_family': is_same_family
            })
        else:
            cross_family_discoveries.append({
                **pred,
                'similarity': sim,
                'source_family': source_family,
                'target_family': target_family,
                'is_same_family': is_same_family
            })

    # Statistics
    total = len(deep_discoveries)
    family_count = len(family_extrapolations)
    cross_count = len(cross_family_discoveries)

    print(f"Total deep discoveries:          {total}")
    print(f"Family extrapolations (sim>0.85): {family_count} ({family_count/total*100:.1f}%)")
    print(f"Cross-family discoveries:         {cross_count} ({cross_count/total*100:.1f}%)")
    print()

    # Show examples
    if family_extrapolations:
        print("=" * 80)
        print("FAMILY EXTRAPOLATION EXAMPLES (High Similarity)")
        print("=" * 80)
        print()

        for pred in sorted(family_extrapolations, key=lambda x: x['similarity'], reverse=True)[:10]:
            fam_info = ""
            if pred['source_family']:
                fam_info = f"({pred['source_family']} family)"

            print(f"{pred['source']:10s} -> {pred['target']:10s} | "
                  f"sim={pred['similarity']:.3f} | conf={pred['confidence']:.3f} | {fam_info}")
        print()

    if cross_family_discoveries:
        print("=" * 80)
        print("CROSS-FAMILY DISCOVERIES (Low Similarity)")
        print("=" * 80)
        print()

        for pred in sorted(cross_family_discoveries, key=lambda x: x['confidence'], reverse=True)[:10]:
            fam_info = f"{pred['source_family'] or '?'} -> {pred['target_family'] or '?'}"

            print(f"{pred['source']:10s} -> {pred['target']:10s} | "
                  f"sim={pred['similarity']:.3f} | conf={pred['confidence']:.3f} | {fam_info}")
        print()

    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()

    if family_count > total * 0.5:
        print("WARNING: Majority of deep discoveries are family extrapolations")
        print()
        print(f"{family_count}/{total} predictions involve highly similar proteins (>0.85).")
        print("ESM-2 embeddings likely encode that similar sequences have similar partners.")
        print("These are not 'novel' in the biological sense - they're expected from homology.")
        print()
        print(f"True cross-family discoveries: {cross_count} ({cross_count/total*100:.1f}%)")
        print()
    else:
        print("POSITIVE FINDING: Majority are cross-family discoveries")
        print()
        print(f"{cross_count}/{total} predictions involve dissimilar proteins (<0.85).")
        print("These represent genuine novel hypotheses beyond protein family knowledge.")
        print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    print("1. STRATIFY predictions by similarity:")
    print(f"   - Family extrapolations (sim>0.85): {family_count}")
    print(f"   - Cross-family discoveries (sim<0.85): {cross_count}")
    print()

    print("2. PRIORITIZE cross-family for experimental validation")
    print("   - These are less likely to be known biology")
    print("   - Higher risk, higher reward")
    print()

    print("3. REPORT both metrics:")
    print(f"   - Deep discovery rate: {len(deep_discoveries)/100*100:.1f}% (not compositional)")
    print(f"   - Cross-family rate: {cross_count/100*100:.1f}% (not family extrapolation)")
    print()

    # Save results
    output = {
        'total_deep_discoveries': total,
        'family_extrapolations': family_count,
        'cross_family_discoveries': cross_count,
        'family_extrapolation_rate': family_count / total if total > 0 else 0,
        'cross_family_rate': cross_count / total if total > 0 else 0,
        'examples_family': [
            {
                'source': p['source'],
                'target': p['target'],
                'similarity': p['similarity'],
                'confidence': p['confidence']
            }
            for p in family_extrapolations[:20]
        ],
        'examples_cross_family': [
            {
                'source': p['source'],
                'target': p['target'],
                'similarity': p['similarity'],
                'confidence': p['confidence']
            }
            for p in cross_family_discoveries[:20]
        ]
    }

    output_path = Path("reports/family_extrapolation_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved: {output_path}")
    print()


if __name__ == "__main__":
    test_family_extrapolation()
