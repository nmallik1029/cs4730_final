#!/usr/bin/env python3
"""
plot_loadbalance.py
Generates bar charts comparing the 4 load balancing strategies
from loadbalance_results.csv.

Outputs:
  - loadbalance_throughput.png
  - loadbalance_latency.png
"""

import csv
import matplotlib.pyplot as plt

rows = []
with open("loadbalance_results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

strategies = [r["strategy"] for r in rows]
throughput = [float(r["throughput_rps"]) for r in rows]
mean_lat   = [float(r["mean_latency_ms"]) for r in rows]
p95_lat    = [float(r["p95_ms"]) for r in rows]
p99_lat    = [float(r["p99_ms"]) for r in rows]

colors = ["#1D9E75", "#378ADD", "#D85A30", "#7F77DD"]

# --- Throughput chart ---
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(strategies, throughput, color=colors, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, throughput):
    ax.text(bar.get_x() + bar.get_width() / 2, val + max(throughput) * 0.01,
            f"{val:.0f}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Throughput (requests/second)", fontsize=11)
ax.set_title("Throughput across load balancing strategies", fontsize=12)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("loadbalance_throughput.png", dpi=150)
print("Wrote loadbalance_throughput.png")

# --- Latency chart (grouped bars: mean / P95 / P99) ---
import numpy as np
x = np.arange(len(strategies))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - width, mean_lat, width, label="Mean",  color="#1D9E75", edgecolor="black", linewidth=0.5)
ax.bar(x,         p95_lat,  width, label="P95",   color="#D85A30", edgecolor="black", linewidth=0.5)
ax.bar(x + width, p99_lat,  width, label="P99",   color="#7F77DD", edgecolor="black", linewidth=0.5)

ax.set_ylabel("Latency (ms)", fontsize=11)
ax.set_title("Latency distribution across load balancing strategies", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("loadbalance_latency.png", dpi=150)
print("Wrote loadbalance_latency.png")

# --- Summary table for report ---
print("\n--- Summary table ---")
print(f"{'Strategy':<20}{'Throughput':>12}{'Mean(ms)':>10}{'P95(ms)':>10}{'P99(ms)':>10}")
for r in rows:
    print(f"{r['strategy']:<20}{float(r['throughput_rps']):>12.1f}"
          f"{float(r['mean_latency_ms']):>10.2f}"
          f"{float(r['p95_ms']):>10.2f}"
          f"{float(r['p99_ms']):>10.2f}")
