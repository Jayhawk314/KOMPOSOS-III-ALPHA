"""
Export predictions from JSON to CSV format for analysis.
"""
import json
import csv
from pathlib import Path

def main():
    # Load predictions from comparison JSON
    json_path = Path('reports/bio_embeddings_comparison.json')

    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        print("Run validate_biological_embeddings.py first")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    print("=" * 80)
    print("EXPORTING PREDICTIONS TO CSV")
    print("=" * 80)
    print()

    # Export text predictions (ALL, not just validated)
    text_csv = Path('reports/text_predictions_top50.csv')
    text_all = data.get('text_all_predictions', data.get('text_validated', []))

    print(f"Text predictions: {len(text_all)}")

    with open(text_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Source', 'Target', 'Confidence', 'System'])
        for i, pred in enumerate(text_all, 1):
            source = pred.get('source', '')
            target = pred.get('target', '')
            confidence = pred.get('confidence', 0.680)
            writer.writerow([i, source, target, confidence, 'Text'])

    print(f"Exported: {text_csv}")

    # Export biological predictions (ALL, not just validated)
    bio_csv = Path('reports/bio_predictions_top50.csv')
    bio_all = data.get('bio_all_predictions', data.get('bio_validated', []))

    print(f"Biological predictions: {len(bio_all)}")

    with open(bio_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Source', 'Target', 'Confidence', 'System'])
        for i, pred in enumerate(bio_all, 1):
            source = pred.get('source', '')
            target = pred.get('target', '')
            confidence = pred.get('confidence', 0.720)
            writer.writerow([i, source, target, confidence, 'Biological'])

    print(f"Exported: {bio_csv}")
    print()
    print("=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
