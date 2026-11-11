# Promoter Negatives Study

How negative-example construction (Recipe C: pair-preserving block shuffle vs Recipe D: matched natural genome) shapes what promoter CNN classifiers learn.

## Layout
- data/raw: downloaded source data (not tracked by git)
- data/processed: generated datasets (not tracked by git)
- recipes: Recipe C and Recipe D generators and matching metrics
- models: CNN training code
- explain: ISM, TF-MoDISco, Tomtom pipeline
- stats: noise floor, gap G, permutation test, TOST, bootstrap
- configs: config.yaml (paths, seeds, thresholds)
- docs: decision rules and notes
- tests: unit tests

## Folder ownership
- recipes/ (Recipe C): TBD
- recipes/ (Recipe D): TBD
- models/, explain/, stats/: TBD

## Rules
- Matching limits, the AUROC gate, the fingerprint threshold and the equivalence margin are frozen in docs/decision_rules.md BEFORE any results are generated.
- Every run records its seed and software versions.