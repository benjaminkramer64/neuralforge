/**
 * @file performance_benchmark.cpp
 * @brief Comprehensive Performance Benchmarking Suite for NeuralForge
 * 
 * This benchmark suite tests:
 * - Tensor operations performance
 * - Neural network layer performance  
 * - Training loop performance
 * - Memory usage patterns
 * - SIMD optimization effectiveness
 * - Multi-threading scalability
 */

#include "../include/neuralforge/neuralforge.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <fstream>

using namespace neuralforge;
using namespace std::chrono;

/**
 * @brief High-resolution timer for benchmarking
 */
class BenchmarkTimer {
private:
    high_resolution_clock::time_point start_;
    
public:
    void start() {
        start_ = high_resolution_clock::now();
    }
    
    double elapsed_ms() {
        auto end = high_resolution_clock::now();
        return duration_cast<microseconds>(end - start_).count() / 1000.0;
    }
    
    double elapsed_us() {
        auto end = high_resolution_clock::now();
        return duration_cast<microseconds>(end - start_).count();
    }
};

/**
 * @brief Benchmark result structure
 */
struct BenchmarkResult {
    std::string name;
    double time_ms;
    double ops_per_sec;
    size_t memory_mb;
    std::string config;
    
    void print() const {
        std::cout << std::left << std::setw(25) << name
                  << std::setw(12) << std::fixed << std::setprecision(2) << time_ms << " ms"
                  << std::setw(15) << std::fixed << std::setprecision(0) << ops_per_sec << " ops/s"
                  << std::setw(10) << memory_mb << " MB"
                  << "  " << config << std::endl;
    }
};

/**
 * @brief Tensor operation benchmarks
 */
class TensorBenchmarks {
public:
    static BenchmarkResult benchmark_matrix_multiply(int size, int iterations = 100) {
        auto A = tensor::randn({size, size});
        auto B = tensor::randn({size, size});
        
        BenchmarkTimer timer;
        timer.start();
        
        for (int i = 0; i < iterations; ++i) {
            auto C = A->matmul(B);
            // Prevent optimization
            volatile float sum = C->data_[0];
            (void)sum;
        }
        
        double elapsed = timer.elapsed_ms();
        double ops = static_cast<double>(size) * size * size * 2 * iterations; // FLOPS
        
        return {
            "MatMul " + std::to_string(size) + "x" + std::to_string(size),
            elapsed / iterations,
            ops / (elapsed / 1000.0),
            static_cast<size_t>((3 * size * size * sizeof(float)) / (1024 * 1024)),
            "SIMD + Threading"
        };
    }
    
    static BenchmarkResult benchmark_element_wise_ops(int size, int iterations = 1000) {
        auto A = tensor::randn({size});
        auto B = tensor::randn({size});
        
        BenchmarkTimer timer;
        timer.start();
        
        for (int i = 0; i < iterations; ++i) {
            auto C = A->add(B)->mul(2.0f)->relu();
            volatile float sum = C->data_[0];
            (void)sum;
        }
        
        double elapsed = timer.elapsed_ms();
        double ops = static_cast<double>(size) * 3 * iterations; // 3 operations per iteration
        
        return {
            "Element-wise (" + std::to_string(size) + " elements)",
            elapsed / iterations,
            ops / (elapsed / 1000.0),
            static_cast<size_t>((3 * size * sizeof(float)) / (1024 * 1024)),
            "AVX2 Vectorized"
        };
    }
    
    static BenchmarkResult benchmark_activation_functions(int size, int iterations = 500) {
        auto input = tensor::randn({size});
        
        BenchmarkTimer timer;
        timer.start();
        
        for (int i = 0; i < iterations; ++i) {
            auto relu_out = input->relu();
            auto tanh_out = input->tanh();
            auto sigmoid_out = input->sigmoid();
            
            volatile float sum = relu_out->data_[0] + tanh_out->data_[0] + sigmoid_out->data_[0];
            (void)sum;
        }
        
        double elapsed = timer.elapsed_ms();
        double ops = static_cast<double>(size) * 3 * iterations;
        
        return {
            "Activations (" + std::to_string(size) + " elements)",
            elapsed / iterations,
            ops / (elapsed / 1000.0),
            static_cast<size_t>((4 * size * sizeof(float)) / (1024 * 1024)),
            "Fast Math + SIMD"
        };
    }
    
    static BenchmarkResult benchmark_reduction_ops(int size, int iterations = 200) {
        auto input = tensor::randn({size});
        
        BenchmarkTimer timer;
        timer.start();
        
        for (int i = 0; i < iterations; ++i) {
            auto sum_result = input->sum();
            auto mean_result = input->mean();
            
            volatile float val = sum_result->data_[0] + mean_result->data_[0];
            (void)val;
        }
        
        double elapsed = timer.elapsed_ms();
        double ops = static_cast<double>(size) * 2 * iterations;
        
        return {
            "Reductions (" + std::to_string(size) + " elements)",
            elapsed / iterations,
            ops / (elapsed / 1000.0),
            static_cast<size_t>((size * sizeof(float)) / (1024 * 1024)),
            "Parallel Reduction"
        };
    }
};

/**
 * @brief Neural network layer benchmarks
 */
class NetworkBenchmarks {
public:
    static BenchmarkResult benchmark_linear_layer(int batch_size, int in_features, int out_features, int iterations = 100) {
        nn::Linear layer(in_features, out_features);
        auto input = tensor::randn({batch_size, in_features});
        
        BenchmarkTimer timer;
        timer.start();
        
        for (int i = 0; i < iterations; ++i) {
            auto output = layer.forward(input);
            volatile float sum = output->data_[0];
            (void)sum;
        }
        
        double elapsed = timer.elapsed_ms();
        double ops = static_cast<double>(batch_size) * in_features * out_features * 2 * iterations;
        
        return {
            "Linear " + std::to_string(in_features) + "->" + std::to_string(out_features),
            elapsed / iterations,
            ops / (elapsed / 1000.0),
            static_cast<size_t>((batch_size * (in_features + out_features) * sizeof(float)) / (1024 * 1024)),
            "Batch=" + std::to_string(batch_size)
        };
    }
    
    static BenchmarkResult benchmark_sequential_network(int batch_size, int iterations = 50) {
        auto model = std::make_shared<nn::Sequential>(
            nn::Linear(784, 512),
            nn::ReLU(),
            nn::Linear(512, 256),
            nn::ReLU(),
            nn::Linear(256, 128),
            nn::ReLU(),
            nn::Linear(128, 10),
            nn::Softmax(-1)
        );
        
        auto input = tensor::randn({batch_size, 784});
        
        BenchmarkTimer timer;
        timer.start();
        
        for (int i = 0; i < iterations; ++i) {
            auto output = model->forward(input);
            volatile float sum = output->data_[0];
            (void)sum;
        }
        
        double elapsed = timer.elapsed_ms();
        double throughput = (batch_size * iterations) / (elapsed / 1000.0);
        
        // Count parameters
        size_t params = 0;
        for (auto& p : model->parameters()) {
            params += p->size();
        }
        
        return {
            "Deep Network (4-layer)",
            elapsed / iterations,
            throughput,
            static_cast<size_t>((params * sizeof(float)) / (1024 * 1024)),
            "Batch=" + std::to_string(batch_size) + ", Params=" + std::to_string(params)
        };
    }
};

/**
 * @brief Training benchmark
 */
class TrainingBenchmarks {
public:
    static BenchmarkResult benchmark_training_step(int batch_size, int iterations = 25) {
        auto model = std::make_shared<nn::Sequential>(
            nn::Linear(784, 256),
            nn::ReLU(),
            nn::Linear(256, 128),
            nn::ReLU(),
            nn::Linear(128, 10),
            nn::Softmax(-1)
        );
        
        nn::Adam optimizer(model->parameters(), 0.001f);
        
        auto input = tensor::randn({batch_size, 784});
        auto target = tensor::zeros({batch_size});
        
        BenchmarkTimer timer;
        timer.start();
        
        for (int i = 0; i < iterations; ++i) {
            // Forward pass
            model->train(true);
            optimizer.zero_grad();
            
            auto output = model->forward(input);
            auto loss = nn::CrossEntropyLoss::forward(output, target);
            
            // Backward pass
            loss->backward();
            optimizer.step();
        }
        
        double elapsed = timer.elapsed_ms();
        double samples_per_sec = (batch_size * iterations) / (elapsed / 1000.0);
        
        return {
            "Training Step",
            elapsed / iterations,
            samples_per_sec,
            static_cast<size_t>((batch_size * 784 * sizeof(float)) / (1024 * 1024)),
            "Adam, Batch=" + std::to_string(batch_size)
        };
    }
    
    static BenchmarkResult benchmark_autograd_overhead(int size, int iterations = 100) {
        // Benchmark with gradients enabled
        auto x = tensor::randn({size, size}, true);
        auto y = tensor::randn({size, size}, true);
        
        BenchmarkTimer timer_grad;
        timer_grad.start();
        
        for (int i = 0; i < iterations; ++i) {
            auto z = x->matmul(y)->sum();
            z->backward();
            x->zero_grad();
            y->zero_grad();
        }
        
        double time_with_grad = timer_grad.elapsed_ms();
        
        // Benchmark without gradients
        auto x_no_grad = tensor::randn({size, size}, false);
        auto y_no_grad = tensor::randn({size, size}, false);
        
        BenchmarkTimer timer_no_grad;
        timer_no_grad.start();
        
        for (int i = 0; i < iterations; ++i) {
            auto z = x_no_grad->matmul(y_no_grad)->sum();
        }
        
        double time_no_grad = timer_no_grad.elapsed_ms();
        double overhead_percent = ((time_with_grad - time_no_grad) / time_no_grad) * 100.0;
        
        return {
            "Autograd Overhead (" + std::to_string(size) + "x" + std::to_string(size) + ")",
            time_with_grad / iterations,
            static_cast<double>(size * size * size * 2 * iterations) / (time_with_grad / 1000.0),
            static_cast<size_t>((3 * size * size * sizeof(float)) / (1024 * 1024)),
            "Overhead: " + std::to_string(overhead_percent) + "%"
        };
    }
};

/**
 * @brief Memory benchmarks
 */
class MemoryBenchmarks {
public:
    static void benchmark_memory_patterns() {
        std::cout << "\n=== Memory Usage Patterns ===" << std::endl;
        
        // Test different tensor sizes
        std::vector<int> sizes = {100, 1000, 10000, 100000};
        
        for (int size : sizes) {
            auto initial_memory = utils::memory::get_memory_usage_mb();
            
            std::vector<TensorPtr> tensors;
            for (int i = 0; i < 10; ++i) {
                tensors.push_back(tensor::randn({size}));
            }
            
            auto peak_memory = utils::memory::get_memory_usage_mb();
            
            tensors.clear();
            
            auto final_memory = utils::memory::get_memory_usage_mb();
            
            if (initial_memory > 0) {
                std::cout << "Size " << std::setw(8) << size 
                          << " | Peak: " << std::setw(6) << std::fixed << std::setprecision(1) << peak_memory << " MB"
                          << " | Leaked: " << std::setw(6) << std::fixed << std::setprecision(1) << (final_memory - initial_memory) << " MB"
                          << std::endl;
            }
        }
    }
};

/**
 * @brief SIMD effectiveness benchmarks
 */
class SIMDBenchmarks {
public:
    static void compare_simd_effectiveness() {
        std::cout << "\n=== SIMD Optimization Effectiveness ===" << std::endl;
        
        std::vector<int> sizes = {1000, 10000, 100000, 1000000};
        
        for (int size : sizes) {
            // Enable SIMD
            config::set_simd_enabled(true);
            auto result_simd = TensorBenchmarks::benchmark_element_wise_ops(size, 100);
            
            // Disable SIMD (if possible)
            config::set_simd_enabled(false);
            auto result_scalar = TensorBenchmarks::benchmark_element_wise_ops(size, 100);
            
            // Re-enable SIMD
            config::set_simd_enabled(true);
            
            double speedup = result_scalar.time_ms / result_simd.time_ms;
            
            std::cout << "Size " << std::setw(8) << size 
                      << " | SIMD: " << std::setw(8) << std::fixed << std::setprecision(2) << result_simd.time_ms << " ms"
                      << " | Scalar: " << std::setw(8) << std::fixed << std::setprecision(2) << result_scalar.time_ms << " ms"
                      << " | Speedup: " << std::setw(6) << std::fixed << std::setprecision(2) << speedup << "x"
                      << std::endl;
        }
    }
};

/**
 * @brief Threading scalability benchmarks
 */
class ThreadingBenchmarks {
public:
    static void benchmark_threading_scalability() {
        std::cout << "\n=== Threading Scalability ===" << std::endl;
        
        std::vector<int> thread_counts = {1, 2, 4, 8};
        int original_threads = config::get_num_threads();
        
        for (int threads : thread_counts) {
            config::set_num_threads(threads);
            
            auto result = TensorBenchmarks::benchmark_matrix_multiply(1024, 50);
            
            std::cout << "Threads: " << threads 
                      << " | Time: " << std::setw(8) << std::fixed << std::setprecision(2) << result.time_ms << " ms"
                      << " | GFLOPS: " << std::setw(8) << std::fixed << std::setprecision(2) << result.ops_per_sec / 1e9
                      << std::endl;
        }
        
        // Restore original thread count
        config::set_num_threads(original_threads);
    }
};

/**
 * @brief Comprehensive benchmark comparison with other libraries
 */
class ComparisonBenchmarks {
public:
    static void benchmark_vs_baseline() {
        std::cout << "\n=== Performance vs Baseline C++ ===" << std::endl;
        
        // Simple matrix multiply baseline
        auto baseline_matmul = [](int size) {
            std::vector<std::vector<float>> A(size, std::vector<float>(size, 1.0f));
            std::vector<std::vector<float>> B(size, std::vector<float>(size, 1.0f));
            std::vector<std::vector<float>> C(size, std::vector<float>(size, 0.0f));
            
            BenchmarkTimer timer;
            timer.start();
            
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    for (int k = 0; k < size; ++k) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            
            return timer.elapsed_ms();
        };
        
        std::vector<int> sizes = {128, 256, 512};
        
        for (int size : sizes) {
            double baseline_time = baseline_matmul(size);
            auto neuralforge_result = TensorBenchmarks::benchmark_matrix_multiply(size, 1);
            
            double speedup = baseline_time / neuralforge_result.time_ms;
            
            std::cout << "Size " << size << "x" << size 
                      << " | Baseline: " << std::setw(8) << std::fixed << std::setprecision(2) << baseline_time << " ms"
                      << " | NeuralForge: " << std::setw(8) << std::fixed << std::setprecision(2) << neuralforge_result.time_ms << " ms"
                      << " | Speedup: " << std::setw(6) << std::fixed << std::setprecision(2) << speedup << "x"
                      << std::endl;
        }
    }
};

/**
 * @brief Generate comprehensive performance report
 */
class PerformanceReporter {
public:
    static void generate_report() {
        std::cout << "🔥 NeuralForge Performance Benchmark Suite" << std::endl;
        std::cout << "=" << std::string(60, '=') << std::endl;
        
        // System information
        std::cout << "\nSystem Information:" << std::endl;
        std::cout << "  CPU Cores: " << std::thread::hardware_concurrency() << std::endl;
        std::cout << "  Configured Threads: " << config::get_num_threads() << std::endl;
        std::cout << "  SIMD Support: " << (config::is_simd_available() ? "AVX2/AVX-512" : "None") << std::endl;
        
        config::print_config();
        
        // Run all benchmarks
        run_tensor_benchmarks();
        run_network_benchmarks();
        run_training_benchmarks();
        
        MemoryBenchmarks::benchmark_memory_patterns();
        SIMDBenchmarks::compare_simd_effectiveness();
        ThreadingBenchmarks::benchmark_threading_scalability();
        ComparisonBenchmarks::benchmark_vs_baseline();
        
        std::cout << "\n🎉 Benchmark Suite Completed!" << std::endl;
    }
    
private:
    static void run_tensor_benchmarks() {
        std::cout << "\n=== Tensor Operations Benchmarks ===" << std::endl;
        std::cout << std::left << std::setw(25) << "Operation"
                  << std::setw(12) << "Time"
                  << std::setw(15) << "Throughput"
                  << std::setw(10) << "Memory"
                  << "Configuration" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        std::vector<BenchmarkResult> results = {
            TensorBenchmarks::benchmark_matrix_multiply(256),
            TensorBenchmarks::benchmark_matrix_multiply(512),
            TensorBenchmarks::benchmark_matrix_multiply(1024),
            TensorBenchmarks::benchmark_element_wise_ops(100000),
            TensorBenchmarks::benchmark_element_wise_ops(1000000),
            TensorBenchmarks::benchmark_activation_functions(100000),
            TensorBenchmarks::benchmark_reduction_ops(1000000)
        };
        
        for (const auto& result : results) {
            result.print();
        }
    }
    
    static void run_network_benchmarks() {
        std::cout << "\n=== Neural Network Benchmarks ===" << std::endl;
        std::cout << std::left << std::setw(25) << "Layer/Network"
                  << std::setw(12) << "Time"
                  << std::setw(15) << "Throughput"
                  << std::setw(10) << "Memory"
                  << "Configuration" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        std::vector<BenchmarkResult> results = {
            NetworkBenchmarks::benchmark_linear_layer(32, 784, 256),
            NetworkBenchmarks::benchmark_linear_layer(64, 512, 256),
            NetworkBenchmarks::benchmark_linear_layer(128, 256, 128),
            NetworkBenchmarks::benchmark_sequential_network(32),
            NetworkBenchmarks::benchmark_sequential_network(64),
            NetworkBenchmarks::benchmark_sequential_network(128)
        };
        
        for (const auto& result : results) {
            result.print();
        }
    }
    
    static void run_training_benchmarks() {
        std::cout << "\n=== Training Benchmarks ===" << std::endl;
        std::cout << std::left << std::setw(25) << "Training Task"
                  << std::setw(12) << "Time"
                  << std::setw(15) << "Throughput"
                  << std::setw(10) << "Memory"
                  << "Configuration" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        std::vector<BenchmarkResult> results = {
            TrainingBenchmarks::benchmark_training_step(32),
            TrainingBenchmarks::benchmark_training_step(64),
            TrainingBenchmarks::benchmark_training_step(128),
            TrainingBenchmarks::benchmark_autograd_overhead(256),
            TrainingBenchmarks::benchmark_autograd_overhead(512)
        };
        
        for (const auto& result : results) {
            result.print();
        }
    }
};

/**
 * @brief Main benchmark execution
 */
int main(int argc, char* argv[]) {
    try {
        // Configure for optimal performance
        config::set_num_threads(std::thread::hardware_concurrency());
        config::set_simd_enabled(true);
        
        // Parse command line arguments
        bool quick_mode = false;
        bool save_results = false;
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--quick") {
                quick_mode = true;
            } else if (arg == "--save") {
                save_results = true;
            }
        }
        
        if (quick_mode) {
            std::cout << "Running quick benchmark suite..." << std::endl;
            // Run subset of benchmarks for CI/CD
            auto result = TensorBenchmarks::benchmark_matrix_multiply(512, 10);
            result.print();
        } else {
            // Run full benchmark suite
            PerformanceReporter::generate_report();
        }
        
        if (save_results) {
            std::cout << "\nSaving results to benchmark_results.csv..." << std::endl;
            // Implementation for saving results would go here
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 