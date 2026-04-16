#!/usr/bin/env bash
# run_loadbalance_experiment.sh
# Runs the load balancing comparison experiment:
#   - For each strategy (round_robin, least_connections, response_time, random)
#   - Starts 4 workers + coordinator on localhost (different ports)
#   - Runs client with N concurrent requests
#   - Parses CSV line from client output
#   - Writes all results to loadbalance_results.csv
#
# Usage:
#   chmod +x run_loadbalance_experiment.sh
#   ./run_loadbalance_experiment.sh
#
# Assumes: coordinator, worker, client executables exist; weights.bin, mnist_test.bin exist.

set -e

COORD_PORT=5000
NUM_REQUESTS=2000
CONCURRENCY=16
STRATEGIES=("round_robin" "least_connections" "response_time" "random")
WORKER_PORTS=(5001 5002 5003 5004)
OUT_FILE="loadbalance_results.csv"

# CSV header
echo "strategy,num_requests,concurrency,errors,wall_sec,throughput_rps,mean_latency_ms,p50_ms,p95_ms,p99_ms,accuracy" > "$OUT_FILE"

cleanup() {
    echo "Cleaning up background processes..."
    pkill -f "./worker " 2>/dev/null || true
    pkill -f "./coordinator " 2>/dev/null || true
    sleep 1
}

trap cleanup EXIT

for strategy in "${STRATEGIES[@]}"; do
    echo "=========================================="
    echo "Testing strategy: $strategy"
    echo "=========================================="

    # Start 4 workers
    for port in "${WORKER_PORTS[@]}"; do
        ./worker "$port" weights.bin > /tmp/worker_$port.log 2>&1 &
    done
    sleep 1  # Give workers time to bind

    # Start coordinator with all 4 workers
    ./coordinator "$COORD_PORT" "$strategy" \
        localhost:5001 localhost:5002 localhost:5003 localhost:5004 \
        > /tmp/coord.log 2>&1 &
    sleep 1  # Let coordinator bind

    # Run client and capture CSV line
    echo "Running $NUM_REQUESTS requests with concurrency=$CONCURRENCY..."
    OUTPUT=$(./client localhost "$COORD_PORT" "$NUM_REQUESTS" "$CONCURRENCY" mnist_test.bin 2>&1)
    CSV_LINE=$(echo "$OUTPUT" | grep "^CSV," | sed 's/^CSV,//')

    if [ -z "$CSV_LINE" ]; then
        echo "ERROR: No CSV line found in output"
        echo "$OUTPUT"
        exit 1
    fi

    # Prepend strategy name to row
    echo "${strategy},${CSV_LINE}" >> "$OUT_FILE"
    echo "Result: ${strategy},${CSV_LINE}"

    # Kill workers and coordinator for next iteration
    cleanup
    sleep 1
done

echo ""
echo "=========================================="
echo "Experiment complete. Results in $OUT_FILE:"
echo "=========================================="
cat "$OUT_FILE"
