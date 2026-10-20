"""Regenerates the E1/E2 paper figures using the corrected gRA estimator from
ra_shift_experiment.py (beta=0, tree pair): Sobol quasi-Monte Carlo with valid
categorical (one-hot) sampling and Wilson confidence intervals, the same
model-agnostic estimator used for every algorithm pair -- instead of the naive
continuous-relaxation Monte Carlo gRA in visualize_rashomon_results.py /
ralign.metrics.geometric.

Source data: results/dt_pruning_rashomon_results_corrected.csv (per-dataset means
over 5 folds, beta=0 subset of results/shift_experiment_raw.csv for
pair=tree_pruned_vs_unpruned).
Output: img/*_corrected.png, same style as the original figures for direct comparison.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("results/dt_pruning_rashomon_results_corrected.csv")

plt.figure(figsize=(5, 3))
sns.histplot(df["gRA"], bins=20, kde=True)
plt.xlabel("gRA")
plt.tight_layout()
plt.savefig("img/hist_gRA_corrected.png")
plt.close()

plt.figure(figsize=(5, 3))
sns.histplot(df["dRA"], bins=20, kde=True)
plt.xlabel("dRA")
plt.tight_layout()
plt.savefig("img/hist_dRA_corrected.png")
plt.close()

plt.figure(figsize=(5, 3))
sns.histplot(df["accdiff"], bins=20, kde=True)
plt.xlabel("Accuracy Difference (Pruned - Unpruned)")
plt.tight_layout()
plt.savefig("img/hist_accdiff_corrected.png")
plt.close()

plt.figure(figsize=(5, 4))
sns.regplot(x="gRA", y="accdiff", data=df)
corr1 = df["gRA"].corr(df["accdiff"])
plt.annotate(f"Correlation: {corr1:.3f}", xy=(0.05, 0.9), xycoords="axes fraction")
plt.tight_layout()
plt.savefig("img/scatter_gRA_accdiff_corrected.png")
plt.close()

plt.figure(figsize=(5, 4))
sns.regplot(x="gRA", y="dRA", data=df)
corr2 = df["gRA"].corr(df["dRA"])
plt.annotate(f"Correlation: {corr2:.3f}", xy=(0.05, 0.9), xycoords="axes fraction")
plt.tight_layout()
plt.savefig("img/scatter_gRA_dRA_corrected.png")
plt.close()

abs_diff = df["accdiff"].abs()
med_gra = df["gRA"].median()
med_diff = abs_diff.median()

colors = ["#e74c3c", "#2ecc71", "#3498db", "#f1c40f"]

fig, ax = plt.subplots(figsize=(6, 5))
ax.axvline(med_gra, color="gray", ls="--", lw=0.8)
ax.axhline(med_diff, color="gray", ls="--", lw=0.8)
quadrant = np.where(
    (abs_diff >= med_diff) & (df["gRA"] < med_gra), 0,
    np.where(
        (abs_diff < med_diff) & (df["gRA"] >= med_gra), 1,
        np.where(
            (abs_diff >= med_diff) & (df["gRA"] >= med_gra), 2, 3,
        ),
    ),
)
q_colors = [colors[q] for q in quadrant]
ax.scatter(df["gRA"], abs_diff, c=q_colors, edgecolor="k", lw=0.3, s=40)
ax.set_xlabel("gRA")
ax.set_ylabel("|Accuracy Difference|")
ax.text(med_gra, ax.get_ylim()[1], "gRA median", ha="center", va="bottom", fontsize=8, color="gray")
ax.text(ax.get_xlim()[1], med_diff, "|Δacc| median", ha="right", va="bottom", fontsize=8, color="gray")

quadrant_labels = [
    "High |Δacc| + Low gRA",
    "Low |Δacc| + High gRA",
    "High |Δacc| + High gRA",
    "Low |Δacc| + Low gRA",
]
handles = [plt.Line2D([0], [0], marker="o", color="w", label=label,
                      markerfacecolor=color, markersize=8)
           for label, color in zip(quadrant_labels, colors)]
ax.legend(handles=handles, title="Quadrant", loc="best")

fig.tight_layout()
fig.savefig("img/quadrants_gRA_accdiff_corrected.png", dpi=150)
plt.close()

print(f"corr(gRA, accdiff) = {corr1:.4f}")
print(f"corr(gRA, dRA)     = {corr2:.4f}")
print(f"median gRA = {med_gra:.4f}, median |accdiff| = {med_diff:.4f}")
for q, label in enumerate(quadrant_labels):
    print(f"{label}: n={int((quadrant==q).sum())}")
