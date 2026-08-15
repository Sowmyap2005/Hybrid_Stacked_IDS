# ==========================================================
# CHART: LATENCY / THROUGHPUT VS BATCH SIZE
# Visualizes the latency benchmark you already ran (Latency.py).
# Uses the numbers you already have - no rerun needed unless
# you want fresh timing numbers.
# ==========================================================

import matplotlib.pyplot as plt
import numpy as np

batch_sizes = [1, 100, 1000, 10000]
per_flow_ms = [212.1267, 2.3595, 0.3454, 0.1181]
throughput = [4.7, 423.8, 2895.5, 8464.2]

fig, ax1 = plt.subplots(figsize=(9, 6))

color1 = '#e74c3c'
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Per-Flow Latency (ms)', color=color1)
ax1.plot(batch_sizes, per_flow_ms, marker='o', color=color1, linewidth=2, label='Per-flow latency (ms)')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
color2 = '#2ecc71'
ax2.set_ylabel('Throughput (flows/sec)', color=color2)
ax2.plot(batch_sizes, throughput, marker='s', color=color2, linewidth=2, label='Throughput (flows/sec)')
ax2.set_yscale('log')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Inference Latency and Throughput vs. Batch Size')
fig.tight_layout()
plt.savefig("latency_throughput_chart.png", dpi=200, bbox_inches="tight")
print("Saved to latency_throughput_chart.png")
