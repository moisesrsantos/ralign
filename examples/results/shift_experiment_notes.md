# Shift-Robustness Experiment: Notes

## Bound validity (delta_gap(beta) = gap(beta) - gap(0), vs TV_hat(beta))
Testing gap(beta) <= TV_hat(beta) directly implicitly assumes gap(0) = 0, which is false in general (dRA and gRA differ even absent any shift). The provable corollary of the dRA*/gRA* TV bound is delta_gap(beta) <= TV_hat(beta).
- Pooled across 6900 (dataset, pair, fold, beta) rows: Pearson r = 0.255 (p=3.7e-103), Spearman rho = 0.320 (p=2.9e-164).
- The bound delta_gap(beta) <= TV_hat(beta) holds in 99.7% of rows (violations beyond this are attributable to gRA/dRA estimation noise).

## Spurious decision flips (models never changed; only the observed test distribution did)
- **tree_pruned_vs_unpruned**: 26/92 datasets (28.3%) show at least one spurious flip across the beta grid at tau=0.7; 92/92 (100.0%) at tau=per-dataset-median-RA.
- **svm_linear_vs_rbf**: 14/92 datasets (15.2%) show at least one spurious flip across the beta grid at tau=0.7; 81/92 (88.0%) at tau=per-dataset-median-RA.
- **logreg_vs_mlp**: 26/92 datasets (28.3%) show at least one spurious flip across the beta grid at tau=0.7; 89/92 (96.7%) at tau=per-dataset-median-RA.

## Cross-pair generality check
- **tree_pruned_vs_unpruned** (gRA method: qmc): mean gap = 0.2307, mean TV_hat = 0.4552, mean gRA = 0.5052 over 92 datasets. Computed with the same model-agnostic QMC estimator used for every pair.
- **svm_linear_vs_rbf** (gRA method: qmc): mean gap = 0.2535, mean TV_hat = 0.4552, mean gRA = 0.6778 over 92 datasets. Computed with the same model-agnostic QMC estimator used for every pair.
- **logreg_vs_mlp** (gRA method: qmc): mean gap = 0.2208, mean TV_hat = 0.4552, mean gRA = 0.6724 over 92 datasets. Computed with the same model-agnostic QMC estimator used for every pair.