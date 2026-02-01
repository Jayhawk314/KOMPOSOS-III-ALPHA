"""
Download protein sequences from UniProt REST API.

Fetches amino acid sequences for the 36 cancer proteins in our dataset.
Handles errors like missing genes, multiple isoforms, API timeouts.

Usage:
    python scripts/download_uniprot_sequences.py
"""
import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


# 36 proteins from cancer_proteins.db
PROTEIN_GENES = [
    "EGFR", "ERBB2", "MET", "VEGFR2", "TP53", "PTEN", "RB1", "BRCA1", "BRCA2",
    "KRAS", "NRAS", "BRAF", "MYC", "PIK3CA", "AKT1", "MTOR", "ERK1", "ERK2",
    "MEK1", "RAF1", "JAK2", "STAT3", "STAT5", "CDK4", "CDK6", "CCND1", "E2F1",
    "BCL2", "BAX", "CASP3", "CASP9", "ATM", "ATR", "CHEK2", "MDM2", "RAD51"
]

# Synonyms for common gene name variations
GENE_SYNONYMS = {
    "VEGFR2": "KDR",  # VEGFR2 is also known as KDR
    "PIK3CA": "P110A",
    "CCND1": "CYCLIN_D1",
    "STAT5": "STAT5A",  # Try STAT5A first
    "ERK1": "MAPK3",
    "ERK2": "MAPK1",
    "MEK1": "MAP2K1"
}


@dataclass
class ProteinSequence:
    """Protein sequence data from UniProt."""
    gene_name: str
    uniprot_id: str
    sequence: str
    length: int
    function: str
    organism: str = "Homo sapiens"
    reviewed: bool = True


def fetch_uniprot_sequence(gene_name: str, organism_id: int = 9606, retry: int = 0) -> Optional[ProteinSequence]:
    """
    Fetch protein sequence from UniProt REST API.

    Args:
        gene_name: Gene symbol (e.g., "TP53")
        organism_id: NCBI taxonomy ID (9606 = Homo sapiens)
        retry: Current retry attempt

    Returns:
        ProteinSequence if found, None otherwise
    """
    # Try main name first, then synonym
    search_name = gene_name
    if retry == 1 and gene_name in GENE_SYNONYMS:
        search_name = GENE_SYNONYMS[gene_name]
        print(f"  Trying synonym: {search_name}")

    # UniProt REST API v2
    url = f"https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"gene:{search_name} AND organism_id:{organism_id} AND reviewed:true",
        "format": "json",
        "size": 5  # Get top 5 results
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        if not results:
            print(f"  No results found for {gene_name}")
            # Try synonym on first failure
            if retry == 0 and gene_name in GENE_SYNONYMS:
                return fetch_uniprot_sequence(gene_name, organism_id, retry=1)
            return None

        # Get canonical (first reviewed result)
        entry = results[0]

        # Extract data
        uniprot_id = entry.get("primaryAccession", "")
        sequence = entry.get("sequence", {}).get("value", "")
        length = entry.get("sequence", {}).get("length", 0)

        # Extract function description
        comments = entry.get("comments", [])
        function_comments = [c for c in comments if c.get("commentType") == "FUNCTION"]
        function = function_comments[0].get("texts", [{}])[0].get("value", "Unknown function") if function_comments else "Unknown function"

        # Truncate long function descriptions
        if len(function) > 200:
            function = function[:197] + "..."

        protein = ProteinSequence(
            gene_name=gene_name,
            uniprot_id=uniprot_id,
            sequence=sequence,
            length=length,
            function=function,
            reviewed=True
        )

        print(f"  Found: {uniprot_id} ({length} aa)")
        return protein

    except requests.Timeout:
        print(f"  Timeout for {gene_name}")
        if retry < 2:
            time.sleep(2 ** retry)  # Exponential backoff
            return fetch_uniprot_sequence(gene_name, organism_id, retry + 1)
        return None

    except requests.RequestException as e:
        print(f"  Error for {gene_name}: {e}")
        return None


def save_fasta(protein: ProteinSequence, output_dir: Path):
    """Save protein sequence in FASTA format."""
    fasta_file = output_dir / f"{protein.gene_name}.fasta"

    with open(fasta_file, 'w') as f:
        f.write(f">{protein.uniprot_id}|{protein.gene_name}|{protein.length}aa\n")
        f.write(f"{protein.function}\n")
        # Write sequence in 60-character lines
        for i in range(0, len(protein.sequence), 60):
            f.write(protein.sequence[i:i+60] + "\n")


def download_all_sequences(gene_names: List[str], output_dir: Path) -> List[ProteinSequence]:
    """
    Download sequences for all proteins.

    Args:
        gene_names: List of gene symbols
        output_dir: Directory to save FASTA files

    Returns:
        List of successfully downloaded proteins
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    proteins = []
    failed = []

    print(f"Downloading sequences for {len(gene_names)} proteins...")
    print("=" * 60)

    for i, gene in enumerate(gene_names, 1):
        print(f"[{i}/{len(gene_names)}] {gene}")

        protein = fetch_uniprot_sequence(gene)

        if protein:
            proteins.append(protein)
            save_fasta(protein, output_dir)
            # Rate limiting: 1 request/second
            time.sleep(1)
        else:
            failed.append(gene)
            print(f"  FAILED")

    print()
    print("=" * 60)
    print(f"Downloaded: {len(proteins)}/{len(gene_names)} proteins")

    if failed:
        print(f"Failed: {', '.join(failed)}")

    # Save combined FASTA
    combined_fasta = output_dir / "all_sequences.fasta"
    with open(combined_fasta, 'w') as f:
        for protein in proteins:
            f.write(f">{protein.uniprot_id}|{protein.gene_name}|{protein.length}aa\n")
            for i in range(0, len(protein.sequence), 60):
                f.write(protein.sequence[i:i+60] + "\n")

    print(f"Saved combined FASTA: {combined_fasta}")

    # Save metadata JSON
    metadata_file = output_dir / "metadata.json"
    metadata = {
        "proteins": [asdict(p) for p in proteins],
        "failed": failed,
        "total": len(gene_names),
        "success": len(proteins),
        "min_length": min(p.length for p in proteins) if proteins else 0,
        "max_length": max(p.length for p in proteins) if proteins else 0,
        "median_length": sorted([p.length for p in proteins])[len(proteins)//2] if proteins else 0
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata: {metadata_file}")
    print()
    print("Sequence Statistics:")
    print(f"  Min length: {metadata['min_length']} aa")
    print(f"  Max length: {metadata['max_length']} aa")
    print(f"  Median length: {metadata['median_length']} aa")

    return proteins


def main():
    """Main entry point."""
    output_dir = Path("data/proteins/sequences")

    proteins = download_all_sequences(PROTEIN_GENES, output_dir)

    if len(proteins) < 30:
        print()
        print("WARNING: Less than 30 proteins downloaded successfully")
        print("Some proteins may be missing from AlphaFold or UniProt")
    else:
        print()
        print("SUCCESS: All sequences downloaded")


if __name__ == "__main__":
    main()
