# Phase 3: Drug Target Mapping - Simplified Plan

## Goal
Identify which of the 93 novel predictions are therapeutically actionable based on existing drugs.

## Why This Matters
- 93% of predictions are novel (not in training data)
- Need to prioritize which ones to validate experimentally
- Focus on targets that ALREADY have drugs → faster to therapy

## Simple 3-Step Approach

### Step 1: Identify Druggable Proteins in Our Dataset (30 min)

**Druggable Classes:**
- Kinases (CDK4, CDK6, BRAF, EGFR, MTOR, PIK3CA, JAK2, etc.) → Inhibitors exist
- Receptors (EGFR, VEGFR2) → Monoclonal antibodies exist
- Proteases (CASP3) → Inhibitors/activators exist
- DNA repair (BRCA1/2, RAD51) → PARP inhibitors synergize
- Oncogenes (MYC, KRAS) → Indirect targeting possible

**Output:** Simple JSON database of our 36 proteins with drug info

```json
{
  "CDK4": {
    "druggable": true,
    "drug_class": "kinase",
    "fda_approved": ["Palbociclib", "Ribociclib", "Abemaciclib"],
    "mechanism": "CDK4/6 inhibitor",
    "cancers": ["breast", "lung"]
  },
  "MYC": {
    "druggable": false,
    "drug_class": "transcription_factor",
    "fda_approved": [],
    "mechanism": "No direct inhibitor (undruggable)",
    "indirect_strategies": ["BET inhibitors", "CDK7 inhibitors"]
  }
}
```

### Step 2: Score Predictions by Therapeutic Potential (15 min)

**Scoring System:**
```
Therapeutic Score = (Druggability × Confidence × Novelty)

Druggability:
- Both proteins druggable: 1.0
- One protein druggable: 0.5
- Neither druggable: 0.1 (still interesting for biology)

Confidence:
- From conjecture engine (0.5-0.74 range)

Novelty:
- Novel: 1.0
- In training: 0.5
```

### Step 3: Generate Ranked Therapeutic Opportunities (15 min)

**Output Format:**

```
TIER 1: Immediate Drug Opportunities (Both proteins druggable)
  1. BRAF -> MYC | Therapeutic Score: 0.729 | Drugs: Vemurafenib (BRAF) + BET inhibitor (MYC)
  2. CDK6 -> JAK2 | Therapeutic Score: 0.718 | Drugs: Palbociclib (CDK6) + Ruxolitinib (JAK2)

TIER 2: Single-Target Opportunities (One protein druggable)
  1. CHEK2 -> MYC | Therapeutic Score: 0.364 | Drug: BET inhibitor (MYC)
  2. MTOR -> CHEK2 | Therapeutic Score: 0.359 | Drug: Everolimus (MTOR)

TIER 3: Research Opportunities (Neither directly druggable, but biologically interesting)
  1. TP53 -> MYC | Therapeutic Score: 0.074 | Strategy: Restore p53 + target MYC
```

---

## Implementation: Single Script

**File:** `scripts/map_drug_targets.py` (~200 lines)

**What it does:**
1. Load novelty analysis results
2. Load simple druggable proteins database (hardcoded JSON)
3. Compute therapeutic scores
4. Generate 3-tier ranked list
5. Export to `reports/therapeutic_opportunities.csv`

**No complex dependencies:**
- No API calls (avoid rate limits, failures)
- No external databases (ChEMBL, DrugBank require licenses)
- Just a simple hardcoded mapping for our 36 proteins

---

## Expected Output

**reports/therapeutic_opportunities.csv:**
```csv
Tier,Rank,Source,Target,Confidence,Novelty,Therapeutic_Score,Source_Drugs,Target_Drugs,Combination_Strategy
1,1,BRAF,MYC,0.729,NOVEL,0.729,"Vemurafenib, Dabrafenib","BET inhibitors","BRAF inhibitor + MYC transcription blocker"
1,2,CDK6,JAK2,0.718,NOVEL,0.718,"Palbociclib, Ribociclib","Ruxolitinib, Fedratinib","CDK4/6 + JAK2 dual inhibition"
2,1,CHEK2,MYC,0.727,NOVEL,0.364,"None (checkpoint kinase)","BET inhibitors","Indirect MYC targeting"
```

**reports/therapeutic_summary.txt:**
```
THERAPEUTIC OPPORTUNITY SUMMARY
================================================================================

Total Novel Predictions: 93
Tier 1 (Both druggable): 15 predictions
Tier 2 (One druggable): 48 predictions
Tier 3 (Research only): 30 predictions

TOP-3 IMMEDIATE OPPORTUNITIES:

1. BRAF → MYC (Biological, conf=0.729)
   - BRAF: FDA-approved inhibitors (Vemurafenib, Dabrafenib)
   - MYC: BET inhibitors in clinical trials
   - Combination Strategy: BRAF inhibition + MYC transcription blockade
   - Cancer Types: Melanoma, lung, colorectal
   - Validation: Co-IP, drug synergy assays

2. CDK6 → JAK2 (Biological, conf=0.718)
   - CDK6: FDA-approved inhibitors (Palbociclib, Ribociclib)
   - JAK2: FDA-approved inhibitors (Ruxolitinib)
   - Combination Strategy: CDK4/6 + JAK2 dual targeting
   - Cancer Types: Leukemia, breast, lymphoma
   - Validation: Drug combination screening

3. PIK3CA → NRAS (Biological, conf=0.718)
   - PIK3CA: FDA-approved inhibitor (Alpelisib)
   - NRAS: MEK inhibitors (Trametinib)
   - Combination Strategy: PI3K + MEK dual inhibition
   - Cancer Types: Breast, melanoma, colorectal
   - Validation: Synergy screening in NRAS-mutant cells
```

---

## Total Time: ~1 hour

**Breakdown:**
- 30 min: Create druggable proteins database (manual curation)
- 15 min: Write scoring script
- 15 min: Run and generate reports

---

## Decision Point

After Phase 3 completes:

**IF** we find ≥5 Tier-1 opportunities:
- Proceed to Phase 4 (Experimental Validation Plan)
- Proceed to Phase 5 (Publication Package)
- Consider scaling to 270 proteins

**IF** we find <5 Tier-1 opportunities:
- Focus on Tier-2 (single-target strategies)
- Still valuable for biology
- May not justify 270-protein scale-up

---

## Key Simplifications

1. **No API calls** - Hardcode drug info for 36 proteins (fast, reliable)
2. **No complex databases** - Simple JSON, easy to verify
3. **Focus on FDA-approved** - Clinical trials data is too noisy
4. **Manual curation** - 36 proteins × 2 min = 72 min one-time effort
5. **Therapeutic score** - Simple formula, easy to explain

This keeps Phase 3 tractable (~1 hour) while providing actionable therapeutic insights.
