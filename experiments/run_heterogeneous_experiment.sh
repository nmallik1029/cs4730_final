#!/usr/bin/env bash

set -e

if [ ! -f data/weights.bin ] || [ ! -f data/mnist_test.bin ] || [ ! -x worker ] || [ ! -x coordinator ] || [ ! -x client ]; then
    echo "missing data files or executables. run 'make' from project root."
    exit 1
fi

mkdir -p results

PORT=5000
N=3000
C=16
STRATS=("round_robin" "least_connections" "response_time" "random")
OUT="results/heterogeneous_results.csv"

echo "strategy,num_requests,concurrency,errors,wall_sec,throughput_rps,mean_latency_ms,p50_ms,p95_ms,p99_ms,accuracy" > "$OUT"

kill_all() {
    pkill -f "./worker " 2>/dev/null || true
    pkill -f "./coordinator " 2>/dev/null || true
    sleep 1
}
trap kill_all EXIT

for s in "${STRATS[@]}"; do
    echo "--- $s (one slow worker) ---"

    ./worker 5001 data/weights.bin > /tmp/w1.log 2>&1 &
    ./worker 5002 data/weights.bin > /tmp/w2.log 2>&1 &
    ./worker 5003 data/weights.bin > /tmp/w3.log 2>&1 &
    ./worker 5004 data/weights.bin 5 > /tmp/w4.log 2>&1 &   # slow one
    sleep 1

    ./coordinator "$PORT" "$s" \
        localhost:5001 localhost:5002 localhost:5003 localhost:5004 \
        > /tmp/coord.log 2>&1 &
    sleep 1

    out=$(./client localhost "$PORT" "$N" "$C" data/mnist_test.bin 2>&1)
    csv=$(echo "$out" | grep "^CSV," | sed 's/^CSV,//')
    if [ -z "$csv" ]; then
        echo "no CSV line"
        echo "$out"
        exit 1
    fi
    echo "${s},${csv}" >> "$OUT"
    echo "done: ${s},${csv}"

    kill_all
    sleep 1
done

echo ""
echo "results:"
cat "$OUT"