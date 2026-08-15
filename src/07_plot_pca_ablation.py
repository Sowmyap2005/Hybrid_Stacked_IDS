# ==========================================================
# CHART: PCA ABLATION (with vs without PCA, per model)
# Visualizes Table 4 in the paper. Uses the numbers you already
# have from full_ablation_study.py - no retraining needed, this
# just draws the chart from those already-known results.
# ==========================================================

import matplotlib.pyplot as plt
import numpy as np

models = ['Random Forest', 'XGBoost', 'DNN']
with_pca = [0.7909, 0.7482, 0.7036]
without_pca = [0.8826, 0.8971, 0.7923]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - width/2, with_pca, width, label='With PCA', color='#e74c3c')
bars2 = ax.bar(x + width/2, without_pca, width, label='Without PCA', color='#2ecc71')

ax.set_ylabel('Macro F1-Score')
ax.set_title('Impact of PCA Dimensionality Reduction on Macro F1')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("pca_ablation_chart.png", dpi=200, bbox_inches="tight")
print("Saved to pca_ablation_chart.png")
