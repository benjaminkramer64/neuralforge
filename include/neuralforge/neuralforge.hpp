/**
 * @file neuralforge.hpp
 * @brief Main header file for NeuralForge - High-Performance Neural Network Library
 * 
 * NeuralForge is a modern C++ neural network library designed for maximum performance
 * and ease of use. It provides:
 * 
 * - SIMD-optimized tensor operations (AVX2/AVX-512)
 * - Automatic differentiation with computation graphs
 * - Modular neural network layers and activations
 * - Advanced optimization algorithms (SGD, Adam, AdamW)
 * - Production-ready training utilities
 * - Zero-copy operations where possible
 * - Thread-safe and memory-efficient design
 * 
 * @version 1.0.0
 * @author NeuralForge Development Team
 * @date 2024
 * 
 * Example usage:
 * @code
 * #include <neuralforge/neuralforge.hpp>
 * using namespace neuralforge;
 * 
 * // Create a deep neural network
 * auto model = std::make_shared<nn::Sequential>(
 *     nn::Linear(784, 256),
 *     nn::ReLU(),
 *     nn::Linear(256, 128), 
 *     nn::ReLU(),
 *     nn::Linear(128, 10),
 *     nn::Softmax(-1)
 * );
 * 
 * // Create tensors
 * auto input = Tensor::randn({32, 784});  // Batch of 32 samples
 * auto target = Tensor::zeros({32});      // Target labels
 * 
 * // Forward pass
 * auto output = model->forward(input);
 * 
 * // Calculate loss
 * auto loss = nn::CrossEntropyLoss::forward(output, target);
 * 
 * // Backward pass
 * loss->backward();
 * 
 * // Optimization step
 * nn::Adam optimizer(model->parameters(), 0.001f);
 * optimizer.step();
 * @endcode
 */

#pragma once

// Version information
#define NEURALFORGE_VERSION_MAJOR 1
#define NEURALFORGE_VERSION_MINOR 0
#define NEURALFORGE_VERSION_PATCH 0
#define NEURALFORGE_VERSION "1.0.0"

// Platform detection
#ifdef _WIN32
    #define NEURALFORGE_PLATFORM_WINDOWS
#elif defined(__APPLE__)
    #define NEURALFORGE_PLATFORM_MACOS
    #include <TargetConditionals.h>
#elif defined(__linux__)
    #define NEURALFORGE_PLATFORM_LINUX
#endif

// Compiler detection
#ifdef __GNUC__
    #define NEURALFORGE_COMPILER_GCC
#elif defined(__clang__)
    #define NEURALFORGE_COMPILER_CLANG
#elif defined(_MSC_VER)
    #define NEURALFORGE_COMPILER_MSVC
#endif

// Feature detection
#ifdef __AVX2__
    #define NEURALFORGE_HAS_AVX2
#endif

#ifdef __AVX512F__
    #define NEURALFORGE_HAS_AVX512
#endif

// Export macros for Windows DLL
#ifdef NEURALFORGE_PLATFORM_WINDOWS
    #ifdef NEURALFORGE_BUILDING_DLL
        #define NEURALFORGE_API __declspec(dllexport)
    #elif defined(NEURALFORGE_USING_DLL)
        #define NEURALFORGE_API __declspec(dllimport)
    #else
        #define NEURALFORGE_API
    #endif
#else
    #define NEURALFORGE_API
#endif

// Force inline macro
#ifdef NEURALFORGE_COMPILER_MSVC
    #define NEURALFORGE_FORCE_INLINE __forceinline
#else
    #define NEURALFORGE_FORCE_INLINE __attribute__((always_inline)) inline
#endif

// Likely/unlikely hints for branch prediction
#ifdef NEURALFORGE_COMPILER_GCC || NEURALFORGE_COMPILER_CLANG
    #define NEURALFORGE_LIKELY(x) __builtin_expect(!!(x), 1)
    #define NEURALFORGE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define NEURALFORGE_LIKELY(x) (x)
    #define NEURALFORGE_UNLIKELY(x) (x)
#endif

// Alignment macro
#ifdef NEURALFORGE_COMPILER_MSVC
    #define NEURALFORGE_ALIGN(n) __declspec(align(n))
#else
    #define NEURALFORGE_ALIGN(n) __attribute__((aligned(n)))
#endif

// Core includes
#include "tensor.hpp"
#include "nn.hpp"

/**
 * @brief Main namespace for all NeuralForge functionality
 */
namespace neuralforge {

/**
 * @brief Version information structure
 */
struct Version {
    static constexpr int major = NEURALFORGE_VERSION_MAJOR;
    static constexpr int minor = NEURALFORGE_VERSION_MINOR;
    static constexpr int patch = NEURALFORGE_VERSION_PATCH;
    static constexpr const char* string = NEURALFORGE_VERSION;
    
    static std::string get_version_string() {
        return std::to_string(major) + "." + 
               std::to_string(minor) + "." + 
               std::to_string(patch);
    }
    
    static std::string get_build_info() {
        std::string info = "NeuralForge v" + get_version_string() + "\n";
        
        #ifdef NEURALFORGE_PLATFORM_WINDOWS
            info += "Platform: Windows\n";
        #elif defined(NEURALFORGE_PLATFORM_MACOS)
            info += "Platform: macOS\n";
        #elif defined(NEURALFORGE_PLATFORM_LINUX)
            info += "Platform: Linux\n";
        #endif
        
        #ifdef NEURALFORGE_COMPILER_GCC
            info += "Compiler: GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "\n";
        #elif defined(NEURALFORGE_COMPILER_CLANG)
            info += "Compiler: Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "\n";
        #elif defined(NEURALFORGE_COMPILER_MSVC)
            info += "Compiler: MSVC " + std::to_string(_MSC_VER) + "\n";
        #endif
        
        info += "Features:\n";
        #ifdef NEURALFORGE_HAS_AVX2
            info += "  - AVX2 SIMD support\n";
        #endif
        #ifdef NEURALFORGE_HAS_AVX512
            info += "  - AVX-512 SIMD support\n";
        #endif
        #ifdef _OPENMP
            info += "  - OpenMP parallelization\n";
        #endif
        #ifdef NEURALFORGE_CUDA_ENABLED
            info += "  - CUDA GPU acceleration\n";
        #endif
        
        return info;
    }
};

/**
 * @brief Configuration and utilities namespace
 */
namespace config {
    /**
     * @brief Global configuration settings
     */
    struct GlobalConfig {
        // Threading
        static inline int num_threads = std::thread::hardware_concurrency();
        static inline bool use_openmp = true;
        
        // SIMD
        static inline bool use_simd = true;
        static inline bool force_avx2 = false;
        static inline bool force_avx512 = false;
        
        // Memory
        static inline size_t tensor_alignment = 32;  // 32-byte aligned for SIMD
        static inline bool use_memory_pool = true;
        
        // Automatic differentiation
        static inline bool enable_grad_check = false;  // For debugging
        static inline bool retain_graph = false;       // For multiple backwards
        
        // Performance
        static inline bool enable_profiling = false;
        static inline bool print_timing = false;
    };
    
    /**
     * @brief Set number of threads for parallel operations
     */
    inline void set_num_threads(int threads) {
        GlobalConfig::num_threads = std::max(1, threads);
        #ifdef _OPENMP
            omp_set_num_threads(GlobalConfig::num_threads);
        #endif
    }
    
    /**
     * @brief Get current number of threads
     */
    inline int get_num_threads() {
        return GlobalConfig::num_threads;
    }
    
    /**
     * @brief Enable/disable SIMD optimizations
     */
    inline void set_simd_enabled(bool enabled) {
        GlobalConfig::use_simd = enabled;
    }
    
    /**
     * @brief Check if SIMD is enabled and available
     */
    inline bool is_simd_available() {
        #ifdef NEURALFORGE_HAS_AVX2
            return GlobalConfig::use_simd;
        #else
            return false;
        #endif
    }
    
    /**
     * @brief Print configuration information
     */
    inline void print_config() {
        std::cout << Version::get_build_info() << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Threads: " << GlobalConfig::num_threads << std::endl;
        std::cout << "  SIMD: " << (GlobalConfig::use_simd ? "enabled" : "disabled") << std::endl;
        std::cout << "  Tensor alignment: " << GlobalConfig::tensor_alignment << " bytes" << std::endl;
        std::cout << "  Memory pool: " << (GlobalConfig::use_memory_pool ? "enabled" : "disabled") << std::endl;
    }
}

/**
 * @brief Utility functions and helpers
 */
namespace utils {
    /**
     * @brief Simple timer for performance measurement
     */
    class Timer {
    private:
        std::chrono::high_resolution_clock::time_point start_time_;
        
    public:
        Timer() : start_time_(std::chrono::high_resolution_clock::now()) {}
        
        void reset() {
            start_time_ = std::chrono::high_resolution_clock::now();
        }
        
        double elapsed_ms() const {
            auto end_time = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time_).count() / 1000.0;
        }
        
        double elapsed_seconds() const {
            return elapsed_ms() / 1000.0;
        }
    };
    
    /**
     * @brief RAII profiler for automatic timing
     */
    class ScopedTimer {
    private:
        std::string name_;
        Timer timer_;
        
    public:
        ScopedTimer(const std::string& name) : name_(name) {}
        
        ~ScopedTimer() {
            if (config::GlobalConfig::print_timing) {
                std::cout << name_ << ": " << timer_.elapsed_ms() << " ms" << std::endl;
            }
        }
    };
    
    /**
     * @brief Memory usage utilities
     */
    namespace memory {
        /**
         * @brief Get current memory usage in MB
         */
        inline double get_memory_usage_mb() {
            #ifdef NEURALFORGE_PLATFORM_LINUX
                std::ifstream status("/proc/self/status");
                std::string line;
                while (std::getline(status, line)) {
                    if (line.substr(0, 6) == "VmRSS:") {
                        std::istringstream iss(line);
                        std::string dummy;
                        int kb;
                        iss >> dummy >> kb;
                        return kb / 1024.0;
                    }
                }
            #endif
            return -1.0;  // Not available
        }
        
        /**
         * @brief Aligned memory allocation
         */
        inline void* aligned_alloc(size_t size, size_t alignment = 32) {
            #ifdef NEURALFORGE_PLATFORM_WINDOWS
                return _aligned_malloc(size, alignment);
            #else
                void* ptr = nullptr;
                if (posix_memalign(&ptr, alignment, size) != 0) {
                    return nullptr;
                }
                return ptr;
            #endif
        }
        
        /**
         * @brief Aligned memory deallocation
         */
        inline void aligned_free(void* ptr) {
            #ifdef NEURALFORGE_PLATFORM_WINDOWS
                _aligned_free(ptr);
            #else
                free(ptr);
            #endif
        }
    }
    
    /**
     * @brief Random number generation utilities
     */
    namespace random {
        /**
         * @brief Global random number generator
         */
        inline std::mt19937& get_rng() {
            static thread_local std::mt19937 gen(std::random_device{}());
            return gen;
        }
        
        /**
         * @brief Set random seed
         */
        inline void set_seed(uint32_t seed) {
            get_rng().seed(seed);
        }
        
        /**
         * @brief Generate random float in [min, max)
         */
        inline float uniform(float min = 0.0f, float max = 1.0f) {
            std::uniform_real_distribution<float> dist(min, max);
            return dist(get_rng());
        }
        
        /**
         * @brief Generate random normal distribution
         */
        inline float normal(float mean = 0.0f, float std = 1.0f) {
            std::normal_distribution<float> dist(mean, std);
            return dist(get_rng());
        }
    }
}

/**
 * @brief Common tensor creation functions (convenience namespace)
 */
namespace tensor {
    using neuralforge::Tensor;
    
    // Re-export common factory functions for convenience
    inline TensorPtr zeros(const Shape& shape, bool requires_grad = false) {
        return Tensor::zeros(shape, requires_grad);
    }
    
    inline TensorPtr ones(const Shape& shape, bool requires_grad = false) {
        return Tensor::ones(shape, requires_grad);
    }
    
    inline TensorPtr randn(const Shape& shape, bool requires_grad = false, 
                          float mean = 0.0f, float std = 1.0f) {
        return Tensor::randn(shape, requires_grad, mean, std);
    }
    
    inline TensorPtr arange(float start, float end, float step = 1.0f, bool requires_grad = false) {
        return Tensor::arange(start, end, step, requires_grad);
    }
    
    inline TensorPtr eye(size_t n, bool requires_grad = false) {
        auto tensor = zeros({n, n}, requires_grad);
        for (size_t i = 0; i < n; ++i) {
            tensor->data_[i * n + i] = 1.0f;
        }
        return tensor;
    }
}

/**
 * @brief Functional interface for neural network operations
 */
namespace F {
    // Activation functions
    inline TensorPtr relu(TensorPtr input) { return input->relu(); }
    inline TensorPtr tanh(TensorPtr input) { return input->tanh(); }
    inline TensorPtr sigmoid(TensorPtr input) { return input->sigmoid(); }
    
    inline TensorPtr softmax(TensorPtr input, int dim = -1) {
        nn::Softmax softmax_layer(dim);
        return softmax_layer.forward(input);
    }
    
    // Loss functions
    inline TensorPtr mse_loss(TensorPtr input, TensorPtr target) {
        return nn::MSELoss::forward(input, target);
    }
    
    inline TensorPtr cross_entropy(TensorPtr input, TensorPtr target) {
        return nn::CrossEntropyLoss::forward(input, target);
    }
    
    inline TensorPtr bce_loss(TensorPtr input, TensorPtr target) {
        return nn::BCELoss::forward(input, target);
    }
    
    // Tensor operations
    inline TensorPtr linear(TensorPtr input, TensorPtr weight, TensorPtr bias = nullptr) {
        auto output = input->matmul(weight->transpose());
        if (bias) {
            output = output->add(bias);
        }
        return output;
    }
}

} // namespace neuralforge

/**
 * @brief Convenience macros for common operations
 */
#define NEURALFORGE_TIMER(name) neuralforge::utils::ScopedTimer _timer(name)
#define NEURALFORGE_PROFILE() NEURALFORGE_TIMER(__FUNCTION__)

/**
 * @brief Version check macro
 */
#define NEURALFORGE_VERSION_CHECK(major, minor, patch) \
    (NEURALFORGE_VERSION_MAJOR > (major) || \
     (NEURALFORGE_VERSION_MAJOR == (major) && NEURALFORGE_VERSION_MINOR > (minor)) || \
     (NEURALFORGE_VERSION_MAJOR == (major) && NEURALFORGE_VERSION_MINOR == (minor) && NEURALFORGE_VERSION_PATCH >= (patch)))

// Include implementation files
#ifdef NEURALFORGE_HEADER_ONLY
    #include "impl/tensor_impl.hpp"
    #include "impl/nn_impl.hpp"
#endif 