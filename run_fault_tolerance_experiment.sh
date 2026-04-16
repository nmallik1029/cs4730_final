#!/usr/bin/env bash
# run_fault_tolerance_experiment.sh
# Demonstrates coordinator handling a worker failure mid-run.
#
# Timeline:
#   t=0s    : 3 workers + coordinator start
#   t=0s    : client begins sending 5000 requests at concurrency 8
#   t=~3s   : kill worker on port 5002 (simulated crash)
#   t=end   : remaining 2 workers should have handled all requests
#
# Output:
#   - fault_tolerance.log  (full logs)
#   - fault_tolerance_timing.csv  (per-request timestamps and latency)

set -e

COORD_PORT=5000
NUM_REQUESTS=5000
CONCURRENCY=8
KILL_AFTER_SEC=3
KILL_PORT=5002

cleanup() {
    pkill -f "./worker " 2>/dev/null || true
    pkill -f "./coordinator " 2>/dev/null || true
    sleep 1
}
trap cleanup EXIT

cleanup  # Start fresh

echo "=== Fault Tolerance Experiment ==="
echo "Starting 3 workers..."
./worker 5001 weights.bin > /tmp/worker_5001.log 2>&1 &
./worker 5002 weights.bin > /tmp/worker_5002.log 2>&1 &
./worker 5003 weights.bin > /tmp/worker_5003.log 2>&1 &
sleep 1

echo "Starting coordinator with round_robin strategy..."
./coordinator "$COORD_PORT" round_robin \
    localhost:5001 localhost:5002 localhost:5003 \
    > fault_tolerance.log 2>&1 &
sleep 1

echo "Client launching ${NUM_REQUESTS} requests, will kill worker 5002 at t=${KILL_AFTER_SEC}s..."

# Schedule the kill in background
(
    sleep "$KILL_AFTER_SEC"
    echo ">>> Killing worker on port $KILL_PORT at $(date +%H:%M:%S)"
    pkill -f "./worker $KILL_PORT " || true
) &

# Run client (blocks until done)
./client localhost "$COORD_PORT" "$NUM_REQUESTS" "$CONCURRENCY" mnist_test.bin > /tmp/client.out 2>&1
echo ""
echo "=== Client output ==="
cat /tmp/client.out
echo ""
echo "=== Coordinator log (failure detection events) ==="
grep -E "unreachable|marking dead|No alive workers" fault_tolerance.log || echo "(no failure events captured)"
echo ""
echo "Full coordinator log in: fault_tolerance.log"
