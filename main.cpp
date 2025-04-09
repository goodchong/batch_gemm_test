#include <iostream>
#include <string>
#include <stdexcept>

#include "utils.hpp"
#include "benchmark_runner.hpp"

int main(int argc, char** argv) {

    // Check for help flag
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // --- Configuration from Arguments ---
    int batch_size     = get_arg(argc, argv, "--batch_size", 64);
    int M              = get_arg(argc, argv, "--m", 128);
    int N              = get_arg(argc, argv, "--n", 128);
    int K              = get_arg(argc, argv, "--k", 128);
    int num_iterations = get_arg(argc, argv, "--iterations", 100);
    std::string precision = get_arg_str(argc, argv, "--precision", "float");

    // Validate inputs
    if (batch_size <= 0 || M <= 0 || N <= 0 || K <= 0 || num_iterations <= 0) {
        std::cerr << "Error: Dimensions (batch_size, m, n, k) and iterations must be positive." << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // --- Run Benchmarks based on precision ---
    try {
        if (precision == "float") {
            run_benchmarks<float>(batch_size, M, N, K, num_iterations);
        } else if (precision == "double") {
            run_benchmarks<double>(batch_size, M, N, K, num_iterations);
        } else {
            std::cerr << "Error: Invalid precision specified. Use 'float' or 'double'." << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
         std::cerr << "Benchmark execution failed: " << e.what() << std::endl;
         return 1;
    }

    return 0;
}
