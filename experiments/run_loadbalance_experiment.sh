#!/usr/bin/env bash
# run all 4 load balancing strategies with 4 identical workers
# run from project root: ./experiments/run_loadbalance_experiment.sh

set -e

if [ ! -f data/weights.bin ] || [ ! -f data/mnist_test.bin ] || [ ! -x worker ] || [ ! -x coordinator ] || [ ! -x client ]; then
    echo "missing data files or executables. run 'make' from project root."
    exit 1
fi

mkdir -p results logs

PORT=5000
N=2000
C=16
STRATS=("round_robin" "least_connections" "response_time" "random")
PORTS=(5001 5002 5003 5004)
OUT="results/loadbalance_results.csv"

echo "strategy,num_requests,concurrency,errors,wall_sec,throughput_rps,mean_latency_ms,p50_ms,p95_ms,p99_ms,accuracy" > "$OUT"

kill_all() {
    pkill -f "./worker " 2>/dev/null || true
    pkill -f "./coordinator " 2>/dev/null || true
    sleep 1
}
trap kill_all EXIT

for s in "${STRATS[@]}"; do
    echo "--- $s ---"

    for p in "${PORTS[@]}"; do
        ./worker "$p" data/weights.bin > logs/w_$p.log 2>&1 &
    done
    sleep 1

    ./coordinator "$PORT" "$s" \
        localhost:5001 localhost:5002 localhost:5003 localhost:5004 \
        > logs/coord.log 2>&1 &
    sleep 1

    out=$(./client localhost "$PORT" "$N" "$C" data/mnist_test.bin 2>&1)
    csv=$(echo "$out" | grep "^CSV," | sed 's/^CSV,//')
    if [ -z "$csv" ]; then
        echo "no CSV line in output"
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