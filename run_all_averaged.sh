#!/usr/bin/env bash
# runs the 2 load balancing experiments 3 times each and averages the results
# this gives us more defensible numbers than a single run

set -e

RUNS=3

run_n_times() {
    local script=$1
    local csv=$2
    local combined="${csv%.csv}_runs.csv"

    echo "--- running $script x$RUNS ---"

    : > "$combined"  # empty file
    local header_written=0

    for i in $(seq 1 $RUNS); do
        echo "  run $i..."
        ./"$script" > /tmp/run_$i.log 2>&1

        if [ $header_written -eq 0 ]; then
            head -n 1 "$csv" | sed 's/^/run,/' > "$combined"
            header_written=1
        fi
        tail -n +2 "$csv" | sed "s/^/$i,/" >> "$combined"
    done

    echo "--- averaging $combined ---"
    python3 average_runs.py "$combined" > "${csv%.csv}_avg.csv"
    cat "${csv%.csv}_avg.csv"
}

run_n_times run_loadbalance_experiment.sh loadbalance_results.csv
echo ""
run_n_times run_heterogeneous_experiment.sh heterogeneous_results.csv

echo ""
echo "done. averaged results in:"
echo "  loadbalance_results_avg.csv"
echo "  heterogeneous_results_avg.csv"
echo "individual runs in:"
echo "  loadbalance_results_runs.csv"
echo "  heterogeneous_results_runs.csv"
