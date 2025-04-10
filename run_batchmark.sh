#!/bin/bash

# --- Configuration ---
EXECUTABLE="./build/batch_matmul_benchmark"
BATCH_SIZE=2000
ITERATIONS=50
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
printf "%-8s %-5s %-5s %-5s %-10s %-18s %-18s %-18s %-18s %-18s\n" "Prec" "M" "N" "K" "BatchSize" "CPU Time (ms)" "GPU H2D (ms)" "GPU Compute (ms)" "GPU D2H (ms)" "GPU Total (ms)"
printf "%-8s %-5s %-5s %-5s %-10s %-18s %-18s %-18s %-18s %-18s\n" "--------" "-----" "-----" "-----" "----------" "------------------" "------------------" "------------------" "------------------" "------------------"# Loop through parameters
for prec in "${PRECISIONS[@]}"; do
    echo "Running: M=N=K=$mkn, Prec=$prec, Batch=$BATCH_SIZE, Iter=$ITERATIONS..."

    for mkn in "${MKN_VALUES[@]}"; do

        # Run the benchmark and capture output (stdout and stderr)
        output=$($EXECUTABLE --m "$mkn" --n "$mkn" --k "$mkn" \
                           --batch_size "$BATCH_SIZE" \
                           --iterations "$ITERATIONS" \
                           --precision "$prec" 2>&1)
        exit_code=$?

        cpu_time="N/A"
        gpu_h2d_time="N/A"
        gpu_compute_time="N/A"
        gpu_d2h_time="N/A"
        gpu_total_time="N/A" # Initialize new variable

        if [ $exit_code -ne 0 ]; then
            echo "  WARNING: Benchmark command failed for M=N=K=$mkn, Prec=$prec (Exit Code: $exit_code)" >&2
            # Optionally print error output: echo "$output" >&2
            cpu_time="ERROR"
            gpu_h2d_time="ERROR"
            gpu_compute_time="ERROR"
            gpu_d2h_time="ERROR"
            gpu_total_time="ERROR"
        else
            # Parse output for CPU time
            cpu_line=$(echo "$output" | grep "CPU (MKL) Average Time:")
            if [ -n "$cpu_line" ]; then
                cpu_time=$(echo "$cpu_line" | awk '{print $5}')
            fi

            # Parse output for GPU times
            if echo "$output" | grep -q "Skipping GPU benchmark"; then
                 gpu_h2d_time="N/A (Skipped)"
                 gpu_compute_time="N/A (Skipped)"
                 gpu_d2h_time="N/A (Skipped)"
                 gpu_total_time="N/A (Skipped)"
            else
                gpu_h2d_line=$(echo "$output" | grep "GPU (cuBLAS) H2D Copy Time:")
                if [ -n "$gpu_h2d_line" ]; then
                    gpu_h2d_time=$(echo "$gpu_h2d_line" | awk '{print $6}')
                fi

                gpu_compute_line=$(echo "$output" | grep "GPU (cuBLAS) Compute Time:")
                if [ -n "$gpu_compute_line" ]; then
                    gpu_compute_time=$(echo "$gpu_compute_line" | awk '{print $5}')
                fi

                gpu_d2h_line=$(echo "$output" | grep "GPU (cuBLAS) D2H Copy Time:")
                if [ -n "$gpu_d2h_line" ]; then
                    gpu_d2h_time=$(echo "$gpu_d2h_line" | awk '{print $6}')
                fi

                gpu_total_line=$(echo "$output" | grep "GPU (cuBLAS) Approx Total:")
                if [ -n "$gpu_total_line" ]; then
                    gpu_total_time=$(echo "$gpu_total_line" | awk '{print $5}')
                fi

            fi
        fi

        # Print table row using printf for alignment
        printf "%-8s %-5d %-5d %-5d %-10d %-18s %-18s %-18s %-18s %-18s\n" \
               "$prec" "$mkn" "$mkn" "$mkn" "$BATCH_SIZE" "$cpu_time" "$gpu_h2d_time" "$gpu_compute_time" "$gpu_d2h_time" "$gpu_total_time"    
     done # End M/N/K loop
done # End Precision loop

printf "\nBenchmark sweep complete.\n"
