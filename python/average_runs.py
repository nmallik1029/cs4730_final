#!/usr/bin/env python3

import sys
import csv
from collections import defaultdict

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))

# group by strategy
groups = defaultdict(list)
for r in rows:
    groups[r["strategy"]].append(r)

# fields to average
num_fields = ["wall_sec", "throughput_rps", "mean_latency_ms",
              "p50_ms", "p95_ms", "p99_ms", "accuracy"]

# print header
fields = ["strategy", "runs", "errors_total"] + num_fields
print(",".join(fields))

# preserve order of first appearance
seen = []
for r in rows:
    if r["strategy"] not in seen:
        seen.append(r["strategy"])

for strat in seen:
    grp = groups[strat]
    n = len(grp)
    errs = sum(int(r["errors"]) for r in grp)
    out = [strat, str(n), str(errs)]
    for f in num_fields:
        vals = [float(r[f]) for r in grp]
        avg = sum(vals) / len(vals)
        out.append(f"{avg:.4f}")
    print(",".join(out))
