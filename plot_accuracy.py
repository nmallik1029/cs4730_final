#!/usr/bin/env python3
"""
plot_accuracy.py
Generates the headline figure for the federated learning experiment:
Worker-alone accuracy vs. Federated accuracy over communication rounds.

Reads accuracy_A.csv and accuracy_B.csv, writes accuracy_plot.png.

Usage:
    python3 plot_accuracy.py
"""

import csv
import matplotlib.pyplot as plt

def read_csv(path):
    rounds, local, fed = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(int(row["round"]))
            local.append(float(row["local_accuracy"]))
            fed.append(float(row["federated_accuracy"]))
    return rounds, local, fed

rounds_a, local_a, fed_a = read_csv("accuracy_A.csv")
rounds_b, local_b, fed_b = read_csv("accuracy_B.csv")

# Prepend round 0 (random init baseline ~10% for MNIST)
rounds = [0] + rounds_a
fed    = [10.0] + fed_a   # both workers have same fed accuracy, so use A's
la     = [10.0] + local_a
lb     = [10.0] + local_b

fig, ax = plt.subplots(figsize=(8, 5))

# Centralized baseline horizontal line
CENTRALIZED = 96.95
ax.axhline(CENTRALIZED, color="gray", linestyle=":", linewidth=1.5,
           label=f"Centralized baseline ({CENTRALIZED:.2f}%)")

# Federated line (thick, prominent)
ax.plot(rounds, fed, color="#1D9E75", linewidth=2.5, marker="o",
        markersize=4, label="Federated (avg)")

# Individual worker lines (dashed, to show plateau)
ax.plot(rounds, la, color="#378ADD", linewidth=1.5, linestyle="--",
        marker="s", markersize=3, label="Worker A only (digits 0-4)")
ax.plot(rounds, lb, color="#D85A30", linewidth=1.5, linestyle="--",
        marker="^", markersize=3, label="Worker B only (digits 5-9)")

ax.set_xlabel("Communication round", fontsize=11)
ax.set_ylabel("Test accuracy on full MNIST (%)", fontsize=11)
ax.set_title("Federated averaging recovers accuracy despite class-partitioned data",
             fontsize=12)
ax.set_ylim(0, 100)
ax.set_xlim(-0.5, max(rounds) + 0.5)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=10)

plt.tight_layout()
plt.savefig("accuracy_plot.png", dpi=150)
print("Wrote accuracy_plot.png")

# Also print a summary for the report
print(f"\n--- Summary for report ---")
print(f"Final federated accuracy:      {fed[-1]:.2f}%")
print(f"Final Worker A local accuracy: {la[-1]:.2f}%")
print(f"Final Worker B local accuracy: {lb[-1]:.2f}%")
print(f"Centralized baseline:          {CENTRALIZED:.2f}%")
print(f"Gap to centralized:            {CENTRALIZED - fed[-1]:+.2f} pp")
print(f"Federated lift over solo:      {fed[-1] - max(la[-1], lb[-1]):+.2f} pp")
