#include "utils.hpp"
#include <vector>
#include <string>
#include <random>
#include <stdexcept> // For std::stoi exception
#include <iostream> // For std::cerr

// Function to initialize matrices with random values
// Explicit template instantiation for float and double
template<typename T>
void initialize_matrices(std::vector<T>& h_A, std::vector<T>& h_B, int batch_size, int M, int N, int K) {
    std::mt19937 rng(12345); // Fixed seed for reproducibility
    std::uniform_real_distribution<T> dist(static_cast<T>(0.0), static_cast<T>(1.0));

    size_t size_A = static_cast<size_t>(batch_size) * M * K;
    size_t size_B = static_cast<size_t>(batch_size) * K * N;

    h_A.resize(size_A);
    h_B.resize(size_B);

    for (size_t i = 0; i < size_A; ++i) {
        h_A[i] = dist(rng);
    }
    for (size_t i = 0; i < size_B; ++i) {
        h_B[i] = dist(rng);
    }
}
// Explicitly instantiate for float and double
template void initialize_matrices<float>(std::vector<float>& h_A, std::vector<float>& h_B, int batch_size, int M, int N, int K);
template void initialize_matrices<double>(std::vector<double>& h_A, std::vector<double>& h_B, int batch_size, int M, int N, int K);


// Simple argument parser
int get_arg(int argc, char** argv, const std::string& flag, int default_val) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == flag) {
            try {
                return std::stoi(argv[i + 1]);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing integer value for " << flag << ": " << argv[i + 1] << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    return default_val;
}

std::string get_arg_str(int argc, char** argv, const std::string& flag, const std::string& default_val) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == flag) {
            return std::string(argv[i + 1]);
        }
    }
    return default_val;
}

void print_usage(const char* prog_name) {
     std::cerr << "Usage: " << prog_name << " [options]\n"
               << "Options:\n"
               << "  --batch_size <int>   Number of matrices in the batch (default: 64)\n"
               << "  --m <int>            Number of rows of matrix A and C (default: 128)\n"
               << "  --n <int>            Number of columns of matrix B and C (default: 128)\n"
               << "  --k <int>            Number of columns of A / rows of B (default: 128)\n"
               << "  --iterations <int>   Number of benchmark iterations (default: 100)\n"
               << "  --precision <string> Data type: float or double (default: float)\n"
               << "  --help               Show this help message\n";
}
