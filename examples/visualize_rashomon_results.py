import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar resultados
df = pd.read_csv("dt_pruning_rashomon_results.csv")

# Histograma gRA
plt.figure(figsize=(5, 3))
sns.histplot(df["gRA"], bins=20, kde=True)
plt.xlabel("gRA")
plt.tight_layout()
plt.savefig("img/hist_gRA.png")
plt.close()

# Histograma dRA
plt.figure(figsize=(5, 3))
sns.histplot(df["dRA"], bins=20, kde=True)
plt.xlabel("dRA")
plt.tight_layout()
plt.savefig("img/hist_dRA.png")
plt.close()

# Histograma diferença de acurácia
plt.figure(figsize=(5, 3))
sns.histplot(df["accdiff"], bins=20, kde=True)
plt.xlabel("Accuracy Difference (Pruned - Unpruned)")
plt.tight_layout()
plt.savefig("img/hist_accdiff.png")
plt.close()

# Scatter gRA vs accdiff
plt.figure(figsize=(5, 4))
sns.regplot(x="gRA", y="accdiff", data=df)
corr1 = df["gRA"].corr(df["accdiff"])
plt.annotate(f"Correlation: {corr1:.3f}", xy=(0.05, 0.9), xycoords="axes fraction")
plt.tight_layout()
plt.savefig("img/scatter_gRA_accdiff.png")
plt.close()

# Scatter gRA vs dRA
plt.figure(figsize=(5, 4))
sns.regplot(x="gRA", y="dRA", data=df)
corr2 = df["gRA"].corr(df["dRA"])
plt.annotate(f"Correlation: {corr2:.3f}", xy=(0.05, 0.9), xycoords="axes fraction")
plt.tight_layout()
plt.savefig("img/scatter_gRA_dRA.png")
plt.close()

# Quadrant scatter: gRA vs |accdiff| (medians as threshold)
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


# Quadrant legend
quadrant_labels = [
    "High |Δacc| + Low gRA",
    "Low |Δacc| + High gRA",
    "High |Δacc| + High gRA",
    "Low |Δacc| + Low gRA",
]
quadrant_colors = ["#e74c3c", "#2ecc71", "#3498db", "#f1c40f"]
handles = [plt.Line2D([0], [0], marker="o", color="w", label=label,
                      markerfacecolor=color, markersize=8)
           for label, color in zip(quadrant_labels, quadrant_colors)]
ax.legend(handles=handles, title="Quadrant", loc="best")

fig.tight_layout()
fig.savefig("img/quadrants_gRA_accdiff.png", dpi=150)
plt.close()
