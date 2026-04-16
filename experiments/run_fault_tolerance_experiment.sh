#!/usr/bin/env bash

set -e

if [ ! -f data/weights.bin ] || [ ! -f data/mnist_test.bin ] || [ ! -x worker ] || [ ! -x coordinator ] || [ ! -x client ]; then
    echo "missing data files or executables. run 'make' from project root."
    exit 1
fi

mkdir -p results

PORT=5000
N=30000
C=16
KILL_AT=3
KILL_PORT=5002

kill_all() {
    pkill -f "./worker " 2>/dev/null || true
    pkill -f "./coordinator " 2>/dev/null || true
    sleep 1
}
trap kill_all EXIT
kill_all

echo "starting 3 workers (2ms delay each)"
./worker 5001 data/weights.bin 2 > /tmp/w1.log 2>&1 &
./worker 5002 data/weights.bin 2 > /tmp/w2.log 2>&1 &
./worker 5003 data/weights.bin 2 > /tmp/w3.log 2>&1 &
sleep 1

echo "starting coordinator"
./coordinator "$PORT" round_robin \
    localhost:5001 localhost:5002 localhost:5003 \
    > results/fault_tolerance.log 2>&1 &
sleep 1

echo "sending $N requests; killing worker $KILL_PORT at t=${KILL_AT}s"

(
    sleep "$KILL_AT"
    echo ""
    echo ">>> killing worker $KILL_PORT at $(date +%H:%M:%S)"
    pkill -f "./worker $KILL_PORT " || true
) &

./client localhost "$PORT" "$N" "$C" data/mnist_test.bin > /tmp/client.out 2>&1

sleep 1
echo ""
echo "client:"
cat /tmp/client.out
echo ""
echo "coordinator log:"
grep -E "down|send fail|recv fail" results/fault_tolerance.log || echo "(nothing)"