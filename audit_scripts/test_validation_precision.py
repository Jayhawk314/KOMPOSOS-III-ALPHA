"""
VALIDATION PRECISION DECOMPOSITION TEST
========================================

Critical question: Are the 3 validated predictions (6% precision) from:
- Compositional leakage (system finding 2-hop paths)
- Deep discovery (system finding truly novel interactions)

This determines if the 6% precision is real or artifactual.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from test_compositional_leakage import CompositionalLeakageDetector


# Validation pairs from validate_biological_embeddings.py
KNOWN_VALIDATIONS = {
    ("KRAS", "MYC"): {
        "validated": True,
        "evidence": "PMID:24954535 - KRAS activates MYC through MAPK/ERK pathway",
        "papers": "50+"
    },
    ("EGFR", "MYC"): {
        "validated": True,
        "evidence": "PMID:15735682 - EGFR signaling upregulates MYC expression",
        "papers": "40+"
    },
    ("PTEN", "BAX"): {
        "validated": True,
        "evidence": "PMID:11836476 - PTEN loss reduces BAX-mediated apoptosis",
        "papers": "30+"
    },
    ("EGFR", "BRAF"): {
        "validated": True,
        "evidence": "PMID:22328973 - EGFR activates BRAF in resistant cancers",
        "papers": "25+"
    },
    ("EGFR", "RAF1"): {
        "validated": True,
        "evidence": "PMID:9430689 - EGFR directly activates RAF1",
        "papers": "60+"
    },
    ("BRCA1", "RAD51"): {
        "validated": True,
        "evidence": "PMID:9751059 - BRCA1 directly interacts with RAD51 for DNA repair",
        "papers": "200+"
    },
    ("PIK3CA", "MYC"): {
        "validated": True,
        "evidence": "PMID:19805105 - PI3K pathway activates MYC transcription",
        "papers": "35+"
    },
    ("NRAS", "MYC"): {
        "validated": True,
        "evidence": "PMID:15735682 - RAS family proteins activate MYC",
        "papers": "30+"
    },
    ("STAT3", "KRAS"): {
        "validated": True,
        "evidence": "PMID:24769394 - STAT3 upregulates KRAS expression",
        "papers": "15+"
    },
    ("RAF1", "TP53"): {
        "validated": True,
        "evidence": "PMID:9769375 - RAF1 phosphorylates and regulates p53",
        "papers": "20+"
    },
    ("BRAF", "TP53"): {
        "validated": True,
        "evidence": "PMID:15520807 - BRAF affects p53 activity via MDM2",
        "papers": "18+"
    },
    ("CDK4", "TP53"): {
        "validated": True,
        "evidence": "PMID:8479518 - CDK4 phosphorylates p53",
        "papers": "40+"
    },
    ("CDK6", "TP53"): {
        "validated": True,
        "evidence": "PMID:10485846 - CDK6 regulates p53 stability",
        "papers": "25+"
    },
}


def load_bio_predictions():
    """Load biological system predictions from validation report."""
    bio_report_path = Path("reports/bio_embeddings_comparison.json")

    if not bio_report_path.exists():
        print(f"ERROR: {bio_report_path} not found")
        print("Run scripts/validate_biological_embeddings.py first")
        return None

    with open(bio_report_path, 'r') as f:
        report = json.load(f)

    # Extract biological predictions
    bio_preds = report.get('bio_all_predictions', [])

    if not bio_preds:
        print("ERROR: No biological predictions in report")
        return None

    return bio_preds


def test_validation_precision():
    """
    Test which validated predictions are compositional vs deep.

    This answers: Is the 6% precision from graph traversal or true discovery?
    """
    print("=" * 80)
    print("VALIDATION PRECISION DECOMPOSITION TEST")
    print("=" * 80)
    print()

    # Load detector
    detector = CompositionalLeakageDetector("data/proteins/cancer_proteins.db")

    # Load biological predictions
    bio_preds = load_bio_predictions()

    if not bio_preds:
        print("Cannot proceed without predictions")
        return

    print(f"Loaded {len(bio_preds)} biological predictions")
    print()

    # Find which predictions matched validation set
    validated_predictions = []

    for pred in bio_preds:
        pair = (pred['source'], pred['target'])
        reverse_pair = (pred['target'], pred['source'])

        if pair in KNOWN_VALIDATIONS or reverse_pair in KNOWN_VALIDATIONS:
            validated_predictions.append(pred)

    print(f"Found {len(validated_predictions)} predictions that match validation set")
    print()

    if len(validated_predictions) == 0:
        print("WARNING: No validated predictions found!")
        print("This might mean the validation logic differs from the report.")
        return

    # Classify each validated prediction
    print("=" * 80)
    print("CLASSIFICATION OF VALIDATED PREDICTIONS")
    print("=" * 80)
    print()

    compositional_hits = []
    deep_discovery_hits = []

    for pred in validated_predictions:
        source = pred['source']
        target = pred['target']
        confidence = pred['confidence']

        classification = detector.classify_prediction(source, target)

        # Get validation evidence
        evidence = KNOWN_VALIDATIONS.get((source, target),
                                         KNOWN_VALIDATIONS.get((target, source), {}))

        is_deep = classification['classification'] == 'TRULY_NOVEL'

        if is_deep:
            deep_discovery_hits.append(pred)
        else:
            compositional_hits.append(pred)

        status = "DEEP DISCOVERY" if is_deep else "COMPOSITIONAL"
        path_str = " -> ".join(classification['path']) if classification['path'] else "None"

        print(f"[{status:16s}] {source:10s} -> {target:10s}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Classification: {classification['classification']}")
        if classification['path']:
            print(f"  Path: {path_str}")
        print(f"  Evidence: {evidence.get('evidence', 'N/A')}")
        print()

    # Statistics
    total_validated = len(validated_predictions)
    compositional_count = len(compositional_hits)
    deep_count = len(deep_discovery_hits)

    print("=" * 80)
    print("PRECISION DECOMPOSITION")
    print("=" * 80)
    print()

    print(f"Total validated predictions:     {total_validated}")
    print(f"  Compositional (leaked):        {compositional_count} ({compositional_count/total_validated*100:.1f}%)")
    print(f"  Deep discoveries (true hits):  {deep_count} ({deep_count/total_validated*100:.1f}%)")
    print()

    # Calculate precision on 50 predictions
    total_bio_preds = len(bio_preds)
    original_precision = total_validated / total_bio_preds
    compositional_precision = compositional_count / total_bio_preds
    deep_precision = deep_count / total_bio_preds

    print(f"Original precision (all validated): {original_precision*100:.1f}% ({total_validated}/{total_bio_preds})")
    print(f"Compositional precision:             {compositional_precision*100:.1f}% ({compositional_count}/{total_bio_preds})")
    print(f"Deep discovery precision:            {deep_precision*100:.1f}% ({deep_count}/{total_bio_preds})")
    print()

    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()

    if compositional_count > deep_count:
        print("CRITICAL FINDING: Majority of validated predictions are COMPOSITIONAL")
        print()
        print(f"{compositional_count}/{total_validated} validated predictions are reachable via graph paths.")
        print("This means the system is being rewarded for doing transitive closure,")
        print("NOT for discovering truly novel biology.")
        print()
        print(f"True discovery precision: {deep_precision*100:.1f}%")
        print()
    elif deep_count > compositional_count:
        print("POSITIVE FINDING: Majority of validated predictions are DEEP DISCOVERIES")
        print()
        print(f"{deep_count}/{total_validated} validated predictions are NOT in training graph.")
        print("This means the system is genuinely discovering novel interactions,")
        print("not just performing graph traversal.")
        print()
        print(f"Deep discovery precision: {deep_precision*100:.1f}%")
        print()
    else:
        print("MIXED FINDING: Equal split between compositional and deep")
        print()
        print("The validation set contains both compositional (expected) and")
        print("deep discovery (novel) predictions.")
        print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    if compositional_count > 0:
        print("1. EXCLUDE compositional predictions from precision calculation")
        print("   - Report deep discovery precision separately")
        print(f"   - Deep precision: {deep_precision*100:.1f}% (on truly independent validation)")
        print()

    print("2. CREATE independent validation set")
    print("   - Use only TRULY_NOVEL validation pairs:")
    for pred in deep_discovery_hits:
        pair = (pred['source'], pred['target'])
        evidence = KNOWN_VALIDATIONS.get(pair, KNOWN_VALIDATIONS.get((pred['target'], pred['source']), {}))
        print(f"     - {pred['source']} -> {pred['target']} | {evidence.get('evidence', '')}")
    print()

    print("3. EXPERIMENTAL VALIDATION required")
    print("   - Co-IP or Y2H for top 10 deep discoveries")
    print("   - Cannot rely on literature-based validation (biased)")
    print()

    # Save results
    output = {
        'total_validated': total_validated,
        'compositional_count': compositional_count,
        'deep_count': deep_count,
        'original_precision': original_precision,
        'compositional_precision': compositional_precision,
        'deep_precision': deep_precision,
        'compositional_hits': [
            {
                'source': p['source'],
                'target': p['target'],
                'confidence': p['confidence'],
                'classification': detector.classify_prediction(p['source'], p['target'])['classification']
            }
            for p in compositional_hits
        ],
        'deep_hits': [
            {
                'source': p['source'],
                'target': p['target'],
                'confidence': p['confidence']
            }
            for p in deep_discovery_hits
        ]
    }

    output_path = Path("reports/validation_precision_decomposition.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved: {output_path}")
    print()


if __name__ == "__main__":
    test_validation_precision()
