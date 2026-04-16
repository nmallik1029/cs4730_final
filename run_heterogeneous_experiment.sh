#!/usr/bin/env bash
# run_heterogeneous_experiment.sh
# Tests all 4 load balancing strategies with a SLOW worker mixed in.
# This exposes the real differences between strategies that hide on uniform workers.
#
# Setup:
#   Worker 5001: normal
#   Worker 5002: normal
#   Worker 5003: normal
#   Worker 5004: SLOW (5ms artificial delay per request = ~200 req/s max for this worker)
#
# Smart strategies (least_connections, response_time) should detect the slow
# worker and route fewer requests there. Round robin and random keep using it
# equally regardless, so P99 latency will be much worse for those.

set -e

COORD_PORT=5000
NUM_REQUESTS=3000
CONCURRENCY=16
STRATEGIES=("round_robin" "least_connections" "response_time" "random")
OUT_FILE="heterogeneous_results.csv"

echo "strategy,num_requests,concurrency,errors,wall_sec,throughput_rps,mean_latency_ms,p50_ms,p95_ms,p99_ms,accuracy" > "$OUT_FILE"

cleanup() {
    pkill -f "./worker " 2>/dev/null || true
    pkill -f "./coordinator " 2>/dev/null || true
    sleep 1
}
trap cleanup EXIT

for strategy in "${STRATEGIES[@]}"; do
    echo "=========================================="
    echo "Testing strategy: $strategy (1 slow worker)"
    echo "=========================================="

    # 3 normal workers
    ./worker 5001 weights.bin     > /tmp/w5001.log 2>&1 &
    ./worker 5002 weights.bin     > /tmp/w5002.log 2>&1 &
    ./worker 5003 weights.bin     > /tmp/w5003.log 2>&1 &
    # 1 slow worker (5ms artificial delay)
    ./worker 5004 weights.bin 5   > /tmp/w5004.log 2>&1 &
    sleep 1

    ./coordinator "$COORD_PORT" "$strategy" \
        localhost:5001 localhost:5002 localhost:5003 localhost:5004 \
        > /tmp/coord.log 2>&1 &
    sleep 1

    OUTPUT=$(./client localhost "$COORD_PORT" "$NUM_REQUESTS" "$CONCURRENCY" mnist_test.bin 2>&1)
    CSV_LINE=$(echo "$OUTPUT" | grep "^CSV," | sed 's/^CSV,//')

    if [ -z "$CSV_LINE" ]; then
        echo "ERROR: No CSV line"
        echo "$OUTPUT"
        exit 1
    fi

    echo "${strategy},${CSV_LINE}" >> "$OUT_FILE"
    echo "Result: ${strategy},${CSV_LINE}"

    cleanup
    sleep 1
done

echo ""
echo "=========================================="
echo "Experiment complete. Results:"
echo "=========================================="
cat "$OUT_FILE"
