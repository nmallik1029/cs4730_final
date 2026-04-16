#!/usr/bin/env python3
# plots throughput + latency bar charts from loadbalance_results.csv

import csv
import numpy as np
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("loadbalance_results.csv")))

strat = [r["strategy"] for r in rows]
tput = [float(r["throughput_rps"]) for r in rows]
mean = [float(r["mean_latency_ms"]) for r in rows]
p95 = [float(r["p95_ms"]) for r in rows]
p99 = [float(r["p99_ms"]) for r in rows]

colors = ["#1D9E75", "#378ADD", "#D85A30", "#7F77DD"]

# throughput
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(strat, tput, color=colors, edgecolor="black", linewidth=0.5)
for b, v in zip(bars, tput):
    ax.text(b.get_x() + b.get_width()/2, v + max(tput)*0.01,
            f"{v:.0f}", ha="center", va="bottom")
ax.set_ylabel("throughput (req/s)")
ax.set_title("Throughput by strategy")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("loadbalance_throughput.png", dpi=150)
print("wrote loadbalance_throughput.png")

# latency
x = np.arange(len(strat))
w = 0.25
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - w, mean, w, label="mean", color="#1D9E75", edgecolor="black", linewidth=0.5)
ax.bar(x, p95, w, label="p95", color="#D85A30", edgecolor="black", linewidth=0.5)
ax.bar(x + w, p99, w, label="p99", color="#7F77DD", edgecolor="black", linewidth=0.5)
ax.set_ylabel("latency (ms)")
ax.set_title("Latency by strategy")
ax.set_xticks(x); ax.set_xticklabels(strat)
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("loadbalance_latency.png", dpi=150)
print("wrote loadbalance_latency.png")

print(f"\n{'strategy':<20}{'rps':>10}{'mean':>8}{'p95':>8}{'p99':>8}")
for r in rows:
    print(f"{r['strategy']:<20}"
          f"{float(r['throughput_rps']):>10.0f}"
          f"{float(r['mean_latency_ms']):>8.2f}"
          f"{float(r['p95_ms']):>8.2f}"
          f"{float(r['p99_ms']):>8.2f}")
