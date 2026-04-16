#!/usr/bin/env bash
# run_fault_tolerance_experiment.sh
# Demonstrates coordinator handling a worker failure mid-run.
#
# To give the experiment enough time for the kill to land during client requests,
# each worker has a 2ms artificial delay per request.
#
# Timeline:
#   t=0s    : 3 workers + coordinator start
#   t=0s    : client begins sending 30000 requests at concurrency 16
#   t=~3s   : kill worker 5002 mid-run
#   t=end   : client should complete all 30000 with minimal/zero errors

set -e

COORD_PORT=5000
NUM_REQUESTS=30000
CONCURRENCY=16
KILL_AFTER_SEC=3
KILL_PORT=5002

cleanup() {
    pkill -f "./worker " 2>/dev/null || true
    pkill -f "./coordinator " 2>/dev/null || true
    sleep 1
}
trap cleanup EXIT
cleanup

echo "=== Fault Tolerance Experiment ==="
echo "Starting 3 workers (each with 2ms artificial delay)..."
./worker 5001 weights.bin 2 > /tmp/worker_5001.log 2>&1 &
./worker 5002 weights.bin 2 > /tmp/worker_5002.log 2>&1 &
./worker 5003 weights.bin 2 > /tmp/worker_5003.log 2>&1 &
sleep 1

echo "Starting coordinator with round_robin strategy..."
./coordinator "$COORD_PORT" round_robin \
    localhost:5001 localhost:5002 localhost:5003 \
    > fault_tolerance.log 2>&1 &
sleep 1

echo "Launching ${NUM_REQUESTS} requests; will kill worker ${KILL_PORT} at t=${KILL_AFTER_SEC}s..."

# Schedule kill in background
(
    sleep "$KILL_AFTER_SEC"
    KILL_TIME=$(date +%H:%M:%S)
    echo ""
    echo ">>> KILLING WORKER ON PORT $KILL_PORT at $KILL_TIME"
    pkill -f "./worker $KILL_PORT " || true
    echo ">>> Worker $KILL_PORT killed"
    echo ""
) &

# Run client (blocks)
./client localhost "$COORD_PORT" "$NUM_REQUESTS" "$CONCURRENCY" mnist_test.bin > /tmp/client.out 2>&1

sleep 1  # let kill message flush

echo ""
echo "=== Client output ==="
cat /tmp/client.out
echo ""
echo "=== Coordinator log: failure detection events ==="
grep -E "unreachable|marking dead|No alive workers|recv failed|send failed" fault_tolerance.log || echo "(no failure events captured)"
echo ""
echo "Full coordinator log in: fault_tolerance.log"