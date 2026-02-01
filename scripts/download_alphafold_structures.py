"""
Download AlphaFold predicted structures from AlphaFold DB.

Fetches PDB files and extracts pLDDT confidence scores for 36 cancer proteins.

Usage:
    python scripts/download_alphafold_structures.py
"""
import requests
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


def load_uniprot_metadata(sequences_dir: Path) -> Dict[str, str]:
    """Load UniProt IDs from downloaded sequences."""
    metadata_file = sequences_dir / "metadata.json"

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_file}\\n"
            "Run download_uniprot_sequences.py first"
        )

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Map gene_name -> uniprot_id
    uniprot_map = {p["gene_name"]: p["uniprot_id"] for p in metadata["proteins"]}
    return uniprot_map


def fetch_alphafold_structure(gene_name: str, uniprot_id: str, output_dir: Path, retry: int = 0) -> Optional[Path]:
    """
    Download AlphaFold predicted structure using API.

    Args:
        gene_name: Gene symbol (e.g., "TP53")
        uniprot_id: UniProt accession (e.g., "P04637")
        output_dir: Directory to save PDB files
        retry: Current retry attempt

    Returns:
        Path to downloaded PDB file, or None if failed
    """
    # Step 1: Query AlphaFold API to get the correct download URL
    api_url = f"https://alphafold.com/api/prediction/{uniprot_id}"

    try:
        print(f"  Querying AlphaFold API...")
        api_response = requests.get(api_url, timeout=30)
        api_response.raise_for_status()

        api_data = api_response.json()

        # API returns a list of isoforms, take the first (canonical)
        if not isinstance(api_data, list) or len(api_data) == 0:
            print(f"  No predictions found in API response")
            return None

        canonical = api_data[0]

        # Extract PDB URL from canonical isoform
        pdb_url = canonical.get("pdbUrl")
        if not pdb_url:
            print(f"  No PDB URL found in API response")
            return None

        # Extract filename from URL
        pdb_filename = pdb_url.split('/')[-1]
        pdb_file = output_dir / pdb_filename

        # Step 2: Download PDB file from the URL provided by API
        print(f"  Downloading from {pdb_url}...")
        pdb_response = requests.get(pdb_url, timeout=30)
        pdb_response.raise_for_status()

        # Save PDB file
        with open(pdb_file, 'wb') as f:
            f.write(pdb_response.content)

        print(f"  Saved: {pdb_file.name} ({len(pdb_response.content)} bytes)")
        return pdb_file

    except requests.HTTPError as e:
        if e.response.status_code == 404:
            print(f"  Not found in AlphaFold DB")
        else:
            print(f"  HTTP error: {e}")
        return None

    except requests.Timeout:
        print(f"  Timeout")
        if retry < 2:
            time.sleep(2 ** retry)
            return fetch_alphafold_structure(gene_name, uniprot_id, output_dir, retry + 1)
        return None

    except json.JSONDecodeError as e:
        print(f"  API returned invalid JSON: {e}")
        return None

    except Exception as e:
        print(f"  Error: {e}")
        return None


def extract_plddt_scores(pdb_file: Path) -> np.ndarray:
    """
    Extract pLDDT confidence scores from AlphaFold PDB file.

    pLDDT scores are stored in the B-factor column of PDB files.
    Values range from 0-100, higher is better.

    Args:
        pdb_file: Path to PDB file

    Returns:
        Array of pLDDT scores (one per residue)
    """
    plddt_scores = []

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                # B-factor is columns 61-66
                try:
                    b_factor = float(line[60:66].strip())
                    # Only record CA atoms (one per residue)
                    atom_name = line[12:16].strip()
                    if atom_name == "CA":
                        plddt_scores.append(b_factor)
                except ValueError:
                    continue

    return np.array(plddt_scores)


@dataclass
class StructuralFeatures:
    """Structural features from AlphaFold prediction."""
    gene_name: str
    uniprot_id: str
    length: int
    mean_plddt: float
    std_plddt: float
    max_plddt: float
    min_plddt: float
    frac_high_conf: float  # Fraction with pLDDT > 90
    frac_low_conf: float   # Fraction with pLDDT < 70


def compute_structural_features(gene_name: str, uniprot_id: str, pdb_file: Path) -> StructuralFeatures:
    """Compute summary features from structure."""
    plddt_scores = extract_plddt_scores(pdb_file)

    return StructuralFeatures(
        gene_name=gene_name,
        uniprot_id=uniprot_id,
        length=len(plddt_scores),
        mean_plddt=float(np.mean(plddt_scores)),
        std_plddt=float(np.std(plddt_scores)),
        max_plddt=float(np.max(plddt_scores)),
        min_plddt=float(np.min(plddt_scores)),
        frac_high_conf=float(np.sum(plddt_scores > 90) / len(plddt_scores)),
        frac_low_conf=float(np.sum(plddt_scores < 70) / len(plddt_scores))
    )


def download_all_structures(uniprot_map: Dict[str, str], output_dir: Path) -> Tuple[List[StructuralFeatures], Dict]:
    """
    Download structures for all proteins.

    Args:
        uniprot_map: Mapping of gene_name -> uniprot_id
        output_dir: Directory to save PDB files

    Returns:
        (list of features, pLDDT scores dict)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    features = []
    plddt_all = {}
    failed = []

    print(f"Downloading structures for {len(uniprot_map)} proteins...")
    print("=" * 60)

    for i, (gene_name, uniprot_id) in enumerate(uniprot_map.items(), 1):
        print(f"[{i}/{len(uniprot_map)}] {gene_name} ({uniprot_id})")

        pdb_file = fetch_alphafold_structure(gene_name, uniprot_id, output_dir)

        if pdb_file and pdb_file.exists():
            try:
                feat = compute_structural_features(gene_name, uniprot_id, pdb_file)
                features.append(feat)

                # Store pLDDT scores
                plddt_scores = extract_plddt_scores(pdb_file)
                plddt_all[gene_name] = plddt_scores

                print(f"  pLDDT: mean={feat.mean_plddt:.1f}, high_conf={feat.frac_high_conf*100:.1f}%")

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                print(f"  Error extracting features: {e}")
                failed.append(gene_name)
        else:
            failed.append(gene_name)

    print()
    print("=" * 60)
    print(f"Downloaded: {len(features)}/{len(uniprot_map)} structures")

    if failed:
        print(f"Failed: {', '.join(failed)}")

    # Save pLDDT scores as numpy archive
    scores_file = output_dir / "plddt_scores.npz"
    np.savez(scores_file, **plddt_all)
    print(f"Saved pLDDT scores: {scores_file}")

    # Save summary JSON
    summary_file = output_dir / "summary.json"
    summary = {
        "features": [asdict(f) for f in features],
        "failed": failed,
        "total": len(uniprot_map),
        "success": len(features),
        "avg_plddt": float(np.mean([f.mean_plddt for f in features])) if features else 0.0,
        "avg_high_conf": float(np.mean([f.frac_high_conf for f in features])) if features else 0.0,
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary: {summary_file}")
    print()
    print("Structure Quality:")
    print(f"  Average pLDDT: {summary['avg_plddt']:.1f}")
    print(f"  High confidence regions: {summary['avg_high_conf']*100:.1f}%")

    return features, plddt_all


def main():
    """Main entry point."""
    sequences_dir = Path("data/proteins/sequences")
    structures_dir = Path("data/proteins/structures")

    # Load UniProt IDs from sequences
    uniprot_map = load_uniprot_metadata(sequences_dir)

    # Download structures
    features, plddt_scores = download_all_structures(uniprot_map, structures_dir)

    if len(features) < 30:
        print()
        print("WARNING: Less than 30 structures downloaded successfully")
        print("Some proteins may not be in AlphaFold DB (unlikely for human proteins)")
    else:
        print()
        print("SUCCESS: All structures downloaded")


if __name__ == "__main__":
    main()
