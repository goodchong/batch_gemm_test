#!/bin/bash

# --- Configuration ---
EXECUTABLE="./build/batch_matmul_benchmark"
BATCH_SIZE=20000
ITERATIONS=1
MKN_VALUES=(10 20 30 60 80 100 150 200 250 300)
PRECISIONS=("float" "double")

# --- Sanity Check ---
if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: Benchmark executable '$EXECUTABLE' not found or not executable." >&2
    echo "Please build the project first (e.g., run 'mkdir build && cd build && cmake .. && make' from the project root)." >&2
    exit 1
fi

# --- Run Benchmarks ---

# Print table header
printf "\n--- Benchmark Results ---\n"
printf "%-8s %-5s %-5s %-5s %-10s %-18s %-18s\n" "Prec" "M" "N" "K" "BatchSize" "CPU Time (ms)" "GPU Time (ms)"
printf "%-8s %-5s %-5s %-5s %-10s %-18s %-18s\n" "--------" "-----" "-----" "-----" "----------" "------------------" "------------------"

# Loop through parameters
for prec in "${PRECISIONS[@]}"; do
    for mkn in "${MKN_VALUES[@]}"; do
        echo "Running: M=N=K=$mkn, Prec=$prec, Batch=$BATCH_SIZE, Iter=$ITERATIONS..."

        # Run the benchmark and capture output (stdout and stderr)
        output=$($EXECUTABLE --m "$mkn" --n "$mkn" --k "$mkn" \
                           --batch_size "$BATCH_SIZE" \
                           --iterations "$ITERATIONS" \
                           --precision "$prec" 2>&1)
        exit_code=$?

        cpu_time="N/A"
        gpu_time="N/A"

        if [ $exit_code -ne 0 ]; then
            echo "  WARNING: Benchmark command failed for M=N=K=$mkn, Prec=$prec (Exit Code: $exit_code)" >&2
            # Optionally print error output: echo "$output" >&2
            cpu_time="ERROR"
            gpu_time="ERROR"
        else
            # Parse output for CPU time
            # Use grep to find the line, then awk to get the 5th field
            cpu_line=$(echo "$output" | grep "CPU (MKL) Average Time:")
            if [ -n "$cpu_line" ]; then
                cpu_time=$(echo "$cpu_line" | awk '{print $5}')
            fi

            # Parse output for GPU time
            gpu_line=$(echo "$output" | grep "GPU (cuBLAS) Average Time:")
            if [ -n "$gpu_line" ]; then
                gpu_time=$(echo "$gpu_line" | awk '{print $5}')
            elif echo "$output" | grep -q "Skipping GPU benchmark"; then
                 gpu_time="N/A (Skipped)" # Mark specifically if skipped
            fi
        fi

        # Print table row using printf for alignment
        printf "%-8s %-5d %-5d %-5d %-10d %-18s %-18s\n" \
               "$prec" "$mkn" "$mkn" "$mkn" "$BATCH_SIZE" "$cpu_time" "$gpu_time"

    done # End M/N/K loop
done # End Precision loop

printf "\nBenchmark sweep complete.\n"
