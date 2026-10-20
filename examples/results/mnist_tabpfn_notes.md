# MNIST capstone case study: notes

## Per-pair gRA / dRA(beta=0)
- **tree_pruned_vs_unpruned**: gRA = 0.194, dRA(beta=0) = 0.218
- **svm_linear_vs_rbf**: gRA = 0.102, dRA(beta=0) = 0.887
- **tabpfn_vs_mlp**: gRA = 0.226, dRA(beta=0) = 0.868

## Bound validity (pooled across pairs and folds, this dataset only)
- n=75, Pearson r=0.204 (p=0.078), bound holds in 100.0% of rows.

## Decision flips (tau=0.7), this single dataset
- **svm_linear_vs_rbf**: flip at tau=0.7 -> False; flip at tau=median -> True
- **tabpfn_vs_mlp**: flip at tau=0.7 -> False; flip at tau=median -> True
- **tree_pruned_vs_unpruned**: flip at tau=0.7 -> False; flip at tau=median -> True