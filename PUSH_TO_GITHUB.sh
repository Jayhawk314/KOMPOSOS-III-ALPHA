#!/bin/bash
# KOMPOSOS-III GitHub Push Script
# Run this to initialize and push to GitHub

echo "========================================"
echo "KOMPOSOS-III GitHub Push"
echo "========================================"
echo ""

# Step 1: Initialize git repository
echo "[1/6] Initializing git repository..."
git init

# Step 2: Add all files (respects .gitignore)
echo "[2/6] Adding files..."
git add .

# Step 3: Check what will be committed
echo "[3/6] Files to be committed:"
git status --short

# Step 4: Verify no personal data
echo "[4/6] Checking for personal paths..."
if git grep "C:\\\\Users\\\\JAMES" >/dev/null 2>&1; then
    echo "ERROR: Personal paths found! Review before pushing."
    exit 1
else
    echo "✓ No personal paths detected"
fi

if git grep "/c/Users/JAMES" >/dev/null 2>&1; then
    echo "ERROR: Personal paths found! Review before pushing."
    exit 1
else
    echo "✓ Clean"
fi

# Step 5: Create commit
echo "[5/6] Creating commit..."
git commit -m "Initial release: KOMPOSOS-III categorical protein interaction discovery

- 93 novel PPI predictions using ESM-2 + category theory
- 21 FDA-approved drug combinations (Tier-1 opportunities)
- 10% validation precision (ahead of literature)
- Hub clustering analysis (90% involve 9 hub proteins)
- Full mathematical framework + experimental roadmap

Intellectual foundations:
- Category theory (Spivak, Gavranović, Schreiber)
- Protein language models (ESM-2, AlphaFold)
- Systems thinking (Daimler, MLST community)

Ready for experimental validation and collaboration.
See ACKNOWLEDGMENTS.md for full attribution."

# Step 6: Add remote and push
echo "[6/6] Adding remote and pushing..."
git remote add origin https://github.com/Jayhawk314/KOMPOSOS-III-ALPHA.git
git branch -M main
git push -u origin main

echo ""
echo "========================================"
echo "✓ Push complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Visit https://github.com/Jayhawk314/KOMPOSOS-III-ALPHA"
echo "2. Make repository public (if currently private)"
echo "3. Add repository description and topics"
echo "4. Check that README.md renders correctly"
echo ""
