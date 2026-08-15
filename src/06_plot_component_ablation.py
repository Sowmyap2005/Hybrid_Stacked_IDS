# ==========================================================
# CHART: COMPONENT ABLATION (leave-one-component-out)
# Visualizes Table 6. Uses numbers already known from your
# full_component_ablation.py run - no retraining needed.
# ==========================================================

import matplotlib.pyplot as plt
import numpy as np

configs = ['Full stack\n(all 4)', 'Minus\nRandom Forest', 'Minus\nXGBoost', 'Minus\nDNN', 'Minus\nAutoencoder']
macro_f1 = [0.9229, 0.9148, 0.8785, 0.9263, 0.9223]

colors = ['#3498db' if c == 'Full stack\n(all 4)' else
          '#e74c3c' if v == min(macro_f1) else
          '#95a5a6' for c, v in zip(configs, macro_f1)]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(configs, macro_f1, color=colors)

ax.axhline(y=0.9229, color='#3498db', linestyle='--', alpha=0.5, linewidth=1, label='Full stack baseline')
ax.set_ylabel('Macro F1-Score')
ax.set_title('Leave-One-Component-Out Ablation: Impact on Macro F1')
ax.set_ylim(0.85, 0.93)
ax.grid(axis='y', alpha=0.3)
ax.legend()

for bar, val in zip(bars, macro_f1):
    ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, val),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("component_ablation_chart.png", dpi=200, bbox_inches="tight")
print("Saved to component_ablation_chart.png")
