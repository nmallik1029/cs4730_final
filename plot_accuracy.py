#!/usr/bin/env python3
# plots federated accuracy over rounds from accuracy_A.csv and accuracy_B.csv

import csv
import matplotlib.pyplot as plt

def load(p):
    rs, loc, fed = [], [], []
    with open(p) as f:
        for row in csv.DictReader(f):
            rs.append(int(row["round"]))
            loc.append(float(row["local_accuracy"]))
            fed.append(float(row["federated_accuracy"]))
    return rs, loc, fed

rA, laA, fedA = load("accuracy_A.csv")
rB, laB, fedB = load("accuracy_B.csv")

# prepend round 0 at 10% (random guess baseline for 10-class problem)
rs  = [0] + rA
fed = [10.0] + fedA
la  = [10.0] + laA
lb  = [10.0] + laB

CENTRALIZED = 96.95

fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(CENTRALIZED, color="gray", linestyle=":",
           label=f"centralized ({CENTRALIZED}%)")
ax.plot(rs, fed, color="#1D9E75", lw=2.5, marker="o", ms=4, label="federated")
ax.plot(rs, la,  color="#378ADD", lw=1.5, ls="--", marker="s", ms=3, label="Worker A only (0-4)")
ax.plot(rs, lb,  color="#D85A30", lw=1.5, ls="--", marker="^", ms=3, label="Worker B only (5-9)")

ax.set_xlabel("round")
ax.set_ylabel("test accuracy (%)")
ax.set_title("Federated averaging vs worker-alone")
ax.set_ylim(0, 100)
ax.set_xlim(-0.5, max(rs) + 0.5)
ax.grid(alpha=0.3)
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig("accuracy_plot.png", dpi=150)
print("wrote accuracy_plot.png")

print(f"\nsummary:")
print(f"final federated: {fed[-1]:.2f}%")
print(f"worker A alone:  {la[-1]:.2f}%")
print(f"worker B alone:  {lb[-1]:.2f}%")
print(f"centralized:     {CENTRALIZED}%")
