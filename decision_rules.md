# Decision rules (DRAFT v0, not frozen until approved and tagged)

Everything below is fixed BEFORE any model results exist. Changes after
confirmatory results are known are logged in docs/deviations.md.

## 1. Data and split
- Positives: 29,597 EPDnew 006 / hg38 windows (601 letters, TSS at index 300).
  One window containing N was dropped.
- Split: splits/human_split_v1.csv (grouped by identical or overlapping windows,
  64/16/20, seed 42). Used because the published Hadad split does not cover all
  promoters (see docs/deviations.md).
- The test set is used only for the AUROC gate and for explanations. It is never
  used for tuning.

## 2. Matching (Recipes C and D versus real promoters)
- GC = fraction of G or C. CpG = count of CG dinucleotides per window.
  Repeat share = fraction of lowercase (soft-masked) letters.
  Zones = index 0-199, 200-400, 401-600.
- Per-promoter caliper for Recipe D pieces:
  |dGC| <= 0.02, |dCpG count| <= max(2, 10% of the promoter's count),
  |d repeat share| <= 0.05, |dGC in each zone| <= 0.05.
- Check 1 passes when, for GC, CpG and repeat share (overall, per zone, and in
  each of 5 promoter-GC quintile bins), real vs C and real vs D both have
  |standardized mean difference| <= 0.05 and KS distance <= 0.05, AND
  - GC/CpG logistic regression test AUROC <= 0.55 (real vs C, real vs D),
  - 16-pair logistic regression test AUROC <= 0.55 (real vs C).
- If Check 1 fails: widen all calipers by 1.5x, at most twice, then stop and
  report the weaker match. No further widening.

## 3. Model gate
- A run is explained only if test AUROC >= 0.80.
- At least 8 eligible runs per recipe are required. If fewer, stop and consult
  supervisors. No hyperparameter changes after seeing test results.
- Architecture and training settings are frozen after the pilot (Check 2) and
  before the 20 main runs.

## 4. Fingerprint test (C vs D, 10 runs, not explained)
- "No reliable signal": mean test AUROC <= 0.55.
- Above 0.55: leftover mismatch. Identify the feature (per-position profiles,
  pair counts) before interpreting the main result, and report it as a limitation.

## 5. Primary decision
- Metric: Jaccard of motif-family sets.
  G = mean within-recipe agreement - mean between-recipe agreement.
- Different: one-sided exact permutation p < 0.05 (all 92,378 splits) AND the
  95% bootstrap CI (over runs) of G has lower end > 0.
- Same: 90% bootstrap CI of G lies inside +/- delta.
- Both Different and Same hold: report "different but within the margin".
- Neither holds: inconclusive.
- delta = 0.5 x SD of within-recipe pairwise Jaccard (the 90 C-C and D-D pairs),
  limited to the range [0.03, 0.10]. delta is computed and committed BEFORE any
  between-recipe agreement is computed.
- Secondary (Spearman of ISM maps): same construction, reported as supporting
  evidence only.

## 6. Exploratory
- H2 (GC-stratified effect) and H3 (Transformer) are exploratory. No multiplicity
  correction is applied, and they are labeled as such.