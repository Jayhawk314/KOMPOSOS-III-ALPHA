"""
Validate biological protein embeddings (ESM-2) vs text embeddings (MPNet).

Compares predictions from:
1. Biological embeddings: ESM-2 650M trained on protein sequences
2. Text embeddings: all-mpnet-base-v2 trained on scientific literature

Tests both on 36-protein cancer dataset, validates against known interactions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, asdict

from data import KomposOSStore
from data.embeddings import EmbeddingsEngine
from data.bio_embeddings import BiologicalEmbeddingsEngine
from oracle import CategoricalOracle
from oracle.conjecture import ConjectureEngine, Conjecture


# Known validations from literature (from validate_36_predictions.py)
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
    ("TP53", "MYC"): {
        "validated": True,
        "evidence": "Science Advances 2023 (PMID:37939186) - TP53 reactivation disrupts MYC transcriptional program",
        "papers": "60+"
    },
    ("BRAF", "MYC"): {
        "validated": True,
        "evidence": "PMID:24934810 - BRAF and MYC synergize in lung tumor development",
        "papers": "25+"
    },
    # NOTE: CHEK2->MYC is predicted at rank 3 but literature shows OPPOSITE direction (MYC->CHEK2)
    # MYC transcriptionally activates CHEK2 (PMID:23269272), not vice versa
    # This is a directional inconsistency - model predicted CHEK2 phosphorylates MYC
}


@dataclass
class ValidationResult:
    """Results from validating predictions against literature."""
    total: int
    validated: int
    precision: float
    validated_pairs: List[Tuple[str, str]]
    failed_pairs: List[Tuple[str, str]]


@dataclass
class ComparisonReport:
    """Comprehensive comparison between biological and text embeddings."""
    bio_precision: float
    text_precision: float
    improvement: float
    bio_total: int
    text_total: int
    bio_unique: int
    text_unique: int
    overlap: int
    bio_validated: List[Tuple[str, str]]
    text_validated: List[Tuple[str, str]]
    bio_only_validated: List[Tuple[str, str]]
    text_only_validated: List[Tuple[str, str]]


def run_biological_pipeline(
    store: KomposOSStore,
    min_confidence: float = 0.5,
    top_k: int = 100
) -> List[Conjecture]:
    """
    Run conjecture engine with biological ESM-2 embeddings.

    Args:
        store: KomposOS data store with 36-protein dataset
        min_confidence: Minimum confidence threshold
        top_k: Number of predictions to generate

    Returns:
        List of predicted conjectures
    """
    print("=" * 80)
    print("BIOLOGICAL EMBEDDINGS PIPELINE (ESM-2)")
    print("=" * 80)
    print()

    # Load biological embeddings engine
    print("Loading ESM-2 model...")
    bio_embeddings = BiologicalEmbeddingsEngine(device='cpu')

    if not bio_embeddings.is_available:
        raise RuntimeError("ESM-2 model not available. Install fair-esm and download sequences.")

    print(f"  Model: {bio_embeddings.model_name}")
    print(f"  Dimension: {bio_embeddings.dimension}d")
    print()

    # Embed all proteins
    print("Embedding 36 proteins with ESM-2...")
    proteins = store.list_objects(limit=100)
    embedded_count = 0

    for protein in proteins:
        try:
            # Generate biological embedding from amino acid sequence
            embedding = bio_embeddings.embed(protein.name, use_cache=True)
            protein.embedding = embedding
            store.add_object(protein)
            embedded_count += 1
            if embedded_count % 10 == 0:
                print(f"  Embedded {embedded_count}/{len(proteins)} proteins...")
        except Exception as e:
            print(f"  WARNING: Could not embed {protein.name}: {e}")

    print(f"  Total embedded: {embedded_count}/{len(proteins)}")
    print()

    # Run conjecture engine
    print("Running CategoricalOracle with biological embeddings...")
    oracle = CategoricalOracle(store, bio_embeddings, min_confidence=min_confidence)
    engine = ConjectureEngine(oracle, semantic_top_k=10)

    result = engine.conjecture(top_k=top_k, min_confidence=min_confidence)

    print(f"  Generated {len(result.conjectures)} predictions")
    print()

    # Show top 5
    print("Top 5 predictions:")
    for i, conj in enumerate(result.conjectures[:5], 1):
        rel = conj.best.predicted_relation if conj.best else "unknown"
        print(f"  {i}. {conj.source:10s} -> {conj.target:10s} [{rel:15s}] conf={conj.top_confidence:.3f}")
    print()

    return result.conjectures


def run_text_baseline(
    store: KomposOSStore,
    min_confidence: float = 0.5,
    top_k: int = 100
) -> List[Conjecture]:
    """
    Run conjecture engine with text embeddings (baseline).

    Args:
        store: KomposOS data store with 36-protein dataset
        min_confidence: Minimum confidence threshold
        top_k: Number of predictions to generate

    Returns:
        List of predicted conjectures
    """
    print("=" * 80)
    print("TEXT EMBEDDINGS BASELINE (MPNet)")
    print("=" * 80)
    print()

    # Load text embeddings engine
    print("Loading all-mpnet-base-v2 model...")
    text_embeddings = EmbeddingsEngine()

    print(f"  Model: all-mpnet-base-v2")
    print(f"  Dimension: {text_embeddings.dimension}d")
    print()

    # Embed all proteins
    print("Embedding 36 proteins with text embeddings...")
    proteins = store.list_objects(limit=100)
    embedded_count = 0

    for protein in proteins:
        try:
            # Generate text embedding from protein name + type
            text = f"{protein.name} {protein.type_name}"
            embedding = text_embeddings.embed(text)
            protein.embedding = embedding
            store.add_object(protein)
            embedded_count += 1
            if embedded_count % 10 == 0:
                print(f"  Embedded {embedded_count}/{len(proteins)} proteins...")
        except Exception as e:
            print(f"  WARNING: Could not embed {protein.name}: {e}")

    print(f"  Total embedded: {embedded_count}/{len(proteins)}")
    print()

    # Run conjecture engine
    print("Running CategoricalOracle with text embeddings...")
    oracle = CategoricalOracle(store, text_embeddings, min_confidence=min_confidence)
    engine = ConjectureEngine(oracle, semantic_top_k=10)

    result = engine.conjecture(top_k=top_k, min_confidence=min_confidence)

    print(f"  Generated {len(result.conjectures)} predictions")
    print()

    # Show top 5
    print("Top 5 predictions:")
    for i, conj in enumerate(result.conjectures[:5], 1):
        rel = conj.best.predicted_relation if conj.best else "unknown"
        print(f"  {i}. {conj.source:10s} -> {conj.target:10s} [{rel:15s}] conf={conj.top_confidence:.3f}")
    print()

    return result.conjectures


def validate_predictions(conjectures: List[Conjecture]) -> ValidationResult:
    """
    Validate predictions against known interactions from literature.

    Args:
        conjectures: List of predicted conjectures

    Returns:
        ValidationResult with precision and validated pairs
    """
    validated_pairs = []
    failed_pairs = []

    for conj in conjectures:
        pair = (conj.source, conj.target)
        reverse_pair = (conj.target, conj.source)

        # Check both directions (protein interactions can be bidirectional)
        if pair in KNOWN_VALIDATIONS and KNOWN_VALIDATIONS[pair]["validated"]:
            validated_pairs.append(pair)
        elif reverse_pair in KNOWN_VALIDATIONS and KNOWN_VALIDATIONS[reverse_pair]["validated"]:
            validated_pairs.append(pair)
        else:
            failed_pairs.append(pair)

    total = len(conjectures)
    validated = len(validated_pairs)
    precision = validated / total if total > 0 else 0.0

    return ValidationResult(
        total=total,
        validated=validated,
        precision=precision,
        validated_pairs=validated_pairs,
        failed_pairs=failed_pairs
    )


def compute_comparison_metrics(
    bio_conjectures: List[Conjecture],
    text_conjectures: List[Conjecture]
) -> ComparisonReport:
    """
    Generate comprehensive comparison between biological and text embeddings.

    Args:
        bio_conjectures: Predictions from biological embeddings
        text_conjectures: Predictions from text embeddings

    Returns:
        ComparisonReport with detailed metrics
    """
    # Validate both
    bio_result = validate_predictions(bio_conjectures)
    text_result = validate_predictions(text_conjectures)

    # Compute overlap
    bio_pairs = {(c.source, c.target) for c in bio_conjectures}
    text_pairs = {(c.source, c.target) for c in text_conjectures}

    overlap = bio_pairs & text_pairs
    bio_only = bio_pairs - text_pairs
    text_only = text_pairs - bio_pairs

    # Find validated predictions unique to each approach
    bio_validated_set = set(bio_result.validated_pairs)
    text_validated_set = set(text_result.validated_pairs)

    bio_only_validated = list(bio_validated_set - text_validated_set)
    text_only_validated = list(text_validated_set - bio_validated_set)

    return ComparisonReport(
        bio_precision=bio_result.precision,
        text_precision=text_result.precision,
        improvement=bio_result.precision - text_result.precision,
        bio_total=bio_result.total,
        text_total=text_result.total,
        bio_unique=len(bio_only),
        text_unique=len(text_only),
        overlap=len(overlap),
        bio_validated=bio_result.validated_pairs,
        text_validated=text_result.validated_pairs,
        bio_only_validated=bio_only_validated,
        text_only_validated=text_only_validated
    )


def print_comparison_report(report: ComparisonReport, bio_conjectures: List[Conjecture] = None, text_conjectures: List[Conjecture] = None):
    """Print detailed comparison report and save all predictions."""
    print()
    print("=" * 80)
    print("BIOLOGICAL vs TEXT EMBEDDINGS COMPARISON")
    print("=" * 80)
    print()

    print("PREDICTION VALIDATION")
    print(f"Biological (ESM-2) Precision:  {report.bio_precision*100:.1f}% ({len(report.bio_validated)}/{report.bio_total})")
    print(f"Text (MPNet) Precision:        {report.text_precision*100:.1f}% ({len(report.text_validated)}/{report.text_total})")
    print(f"Improvement:                   {report.improvement*100:+.1f}%")
    print()

    print("NOVELTY & COVERAGE")
    print(f"Biological unique predictions: {report.bio_unique}")
    print(f"Text unique predictions:       {report.text_unique}")
    print(f"Predictions by both:           {report.overlap}")
    print()

    print("VALIDATED DISCOVERIES")
    if report.bio_only_validated:
        print(f"Validated ONLY by biological embeddings: {len(report.bio_only_validated)}")
        for pair in report.bio_only_validated:
            evidence = KNOWN_VALIDATIONS.get(pair, KNOWN_VALIDATIONS.get((pair[1], pair[0]), {}))
            print(f"  {pair[0]:10s} -> {pair[1]:10s} | {evidence.get('evidence', 'N/A')}")
    else:
        print("Validated ONLY by biological embeddings: 0")

    print()

    if report.text_only_validated:
        print(f"Validated ONLY by text embeddings: {len(report.text_only_validated)}")
        for pair in report.text_only_validated:
            evidence = KNOWN_VALIDATIONS.get(pair, KNOWN_VALIDATIONS.get((pair[1], pair[0]), {}))
            print(f"  {pair[0]:10s} -> {pair[1]:10s} | {evidence.get('evidence', 'N/A')}")
    else:
        print("Validated ONLY by text embeddings: 0")

    print()
    print("=" * 80)
    print("ASSESSMENT")
    print("=" * 80)
    print()

    if report.improvement > 0.05:
        print("BIOLOGICAL EMBEDDINGS SUPERIOR")
        print(f"  ESM-2 improves precision by {report.improvement*100:.1f}%")
        print("  Recommendation: Use biological embeddings for production")
    elif report.improvement < -0.05:
        print("TEXT EMBEDDINGS SUPERIOR")
        print(f"  MPNet outperforms ESM-2 by {abs(report.improvement)*100:.1f}%")
        print("  Recommendation: Keep using text embeddings OR try hybrid approach")
    else:
        print("COMPARABLE PERFORMANCE")
        print("  Both approaches achieve similar precision")
        print("  Recommendation: Consider hybrid (concatenate embeddings)")

    print()

    # Save report
    report_file = Path("reports/bio_embeddings_comparison.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, 'w') as f:
        # Convert to dict for JSON serialization
        report_dict = {
            "bio_precision": report.bio_precision,
            "text_precision": report.text_precision,
            "improvement": report.improvement,
            "bio_total": report.bio_total,
            "text_total": report.text_total,
            "bio_unique": report.bio_unique,
            "text_unique": report.text_unique,
            "overlap": report.overlap,
            "bio_validated": [{"source": p[0], "target": p[1]} for p in report.bio_validated],
            "text_validated": [{"source": p[0], "target": p[1]} for p in report.text_validated],
            "bio_only_validated": [{"source": p[0], "target": p[1]} for p in report.bio_only_validated],
            "text_only_validated": [{"source": p[0], "target": p[1]} for p in report.text_only_validated],
        }

        # Add ALL predictions (for novelty analysis)
        if bio_conjectures:
            report_dict["bio_all_predictions"] = [
                {
                    "source": c.source,
                    "target": c.target,
                    "confidence": c.top_confidence,
                    "relation": c.best.predicted_relation if c.best else "unknown"
                }
                for c in bio_conjectures
            ]

        if text_conjectures:
            report_dict["text_all_predictions"] = [
                {
                    "source": c.source,
                    "target": c.target,
                    "confidence": c.top_confidence,
                    "relation": c.best.predicted_relation if c.best else "unknown"
                }
                for c in text_conjectures
            ]

        json.dump(report_dict, f, indent=2)

    print(f"Detailed report saved: {report_file}")
    print()


def main():
    """Main entry point."""
    # Load 36-protein cancer dataset
    store = KomposOSStore('data/proteins/cancer_proteins.db')

    print()
    print("=" * 80)
    print("BIOLOGICAL EMBEDDINGS VALIDATION")
    print("Comparing ESM-2 (protein sequences) vs MPNet (text)")
    print("=" * 80)
    print()

    # Run biological pipeline
    bio_conjectures = run_biological_pipeline(store, min_confidence=0.5, top_k=50)

    # Run text baseline
    text_conjectures = run_text_baseline(store, min_confidence=0.5, top_k=50)

    # Compare results
    report = compute_comparison_metrics(bio_conjectures, text_conjectures)

    # Print report with all predictions
    print_comparison_report(report, bio_conjectures, text_conjectures)


if __name__ == "__main__":
    main()
