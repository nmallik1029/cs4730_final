#!/usr/bin/env bash
# same as the load balancing experiment but one worker is slow
# (5ms delay added) so we can actually see strategy differences

set -e

PORT=5000
N=3000
C=16
STRATS=("round_robin" "least_connections" "response_time" "random")
OUT="heterogeneous_results.csv"

echo "strategy,num_requests,concurrency,errors,wall_sec,throughput_rps,mean_latency_ms,p50_ms,p95_ms,p99_ms,accuracy" > "$OUT"

kill_all() {
    pkill -f "./worker " 2>/dev/null || true
    pkill -f "./coordinator " 2>/dev/null || true
    sleep 1
}
trap kill_all EXIT

for s in "${STRATS[@]}"; do
    echo "--- $s (one slow worker) ---"

    ./worker 5001 weights.bin > /tmp/w1.log 2>&1 &
    ./worker 5002 weights.bin > /tmp/w2.log 2>&1 &
    ./worker 5003 weights.bin > /tmp/w3.log 2>&1 &
    ./worker 5004 weights.bin 5 > /tmp/w4.log 2>&1 &   # slow one
    sleep 1

    ./coordinator "$PORT" "$s" \
        localhost:5001 localhost:5002 localhost:5003 localhost:5004 \
        > /tmp/coord.log 2>&1 &
    sleep 1

    out=$(./client localhost "$PORT" "$N" "$C" mnist_test.bin 2>&1)
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
