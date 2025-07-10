/**
 * @file simple_demo.cpp
 * @brief Simple demonstration of NeuralForge capabilities
 * 
 * This demo showcases:
 * - Basic tensor operations
 * - Neural network construction
 * - Training a simple model
 * - Performance features
 */

#include "../include/neuralforge/neuralforge.hpp"
#include <iostream>
#include <iomanip>

using namespace neuralforge;

void demo_tensor_operations() {
    std::cout << "\n🔥 Tensor Operations Demo" << std::endl;
    std::cout << std::string(40, '=') << std::endl;
    
    // Create tensors
    auto a = tensor::randn({3, 4});
    auto b = tensor::ones({3, 4});
    
    std::cout << "Tensor A (random):\n" << *a << std::endl;
    std::cout << "Tensor B (ones):\n" << *b << std::endl;
    
    // Basic operations
    auto c = a->add(b);
    auto d = a->mul(b);
    auto e = a->relu();
    
    std::cout << "A + B:\n" << *c << std::endl;
    std::cout << "A * B:\n" << *d << std::endl;
    std::cout << "ReLU(A):\n" << *e << std::endl;
    
    // Matrix operations
    auto matrix_a = tensor::randn({2, 3});
    auto matrix_b = tensor::randn({3, 2});
    auto matrix_c = matrix_a->matmul(matrix_b);
    
    std::cout << "\nMatrix Multiplication Demo:" << std::endl;
    std::cout << "A (2x3) @ B (3x2) = C (2x2):\n" << *matrix_c << std::endl;
}

void demo_automatic_differentiation() {
    std::cout << "\n🧠 Automatic Differentiation Demo" << std::endl;
    std::cout << std::string(40, '=') << std::endl;
    
    // Create tensors with gradient tracking
    auto x = tensor::randn({2, 2}, true);  // requires_grad=true
    auto y = tensor::randn({2, 2}, true);
    
    std::cout << "Input X:\n" << *x << std::endl;
    std::cout << "Input Y:\n" << *y << std::endl;
    
    // Forward pass: z = (x * y).sum()
    auto z = x->mul(y)->sum();
    std::cout << "Z = sum(X * Y): " << z->data_[0] << std::endl;
    
    // Backward pass
    z->backward();
    
    std::cout << "Gradient of X:\n" << *x->grad_ << std::endl;
    std::cout << "Gradient of Y:\n" << *y->grad_ << std::endl;
}

void demo_neural_network() {
    std::cout << "\n🤖 Neural Network Demo" << std::endl;
    std::cout << std::string(40, '=') << std::endl;
    
    // Create a simple neural network
    auto model = std::make_shared<nn::Sequential>(
        nn::Linear(4, 8),
        nn::ReLU(),
        nn::Linear(8, 4),
        nn::ReLU(),
        nn::Linear(4, 2),
        nn::Softmax(-1)
    );
    
    std::cout << "Model architecture:\n" << model->name() << std::endl;
    
    // Count parameters
    auto params = model->parameters();
    size_t total_params = 0;
    for (const auto& param : params) {
        total_params += param->size();
    }
    std::cout << "Total parameters: " << total_params << std::endl;
    
    // Create sample data
    auto input = tensor::randn({5, 4});  // Batch of 5 samples
    auto target = tensor::zeros({5});    // Target classes
    
    // Fill target with random class labels (0 or 1)
    for (size_t i = 0; i < 5; ++i) {
        target->data_[i] = static_cast<float>(i % 2);
    }
    
    std::cout << "\nInput batch shape: [" << input->shape()[0] << ", " << input->shape()[1] << "]" << std::endl;
    std::cout << "Target classes: ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << static_cast<int>(target->data_[i]) << " ";
    }
    std::cout << std::endl;
    
    // Forward pass
    auto output = model->forward(input);
    std::cout << "\nModel output shape: [" << output->shape()[0] << ", " << output->shape()[1] << "]" << std::endl;
    
    // Calculate loss
    auto loss = F::cross_entropy(output, target);
    std::cout << "Cross-entropy loss: " << loss->data_[0] << std::endl;
}

void demo_training_loop() {
    std::cout << "\n🎯 Training Loop Demo" << std::endl;
    std::cout << std::string(40, '=') << std::endl;
    
    // Create model
    auto model = std::make_shared<nn::Sequential>(
        nn::Linear(2, 4),
        nn::Tanh(),
        nn::Linear(4, 1)
    );
    
    // Create optimizer
    nn::Adam optimizer(model->parameters(), 0.01f);
    
    std::cout << "Training a simple regression model..." << std::endl;
    std::cout << "Target function: y = x1^2 + x2^2" << std::endl;
    
    // Training loop
    for (int epoch = 1; epoch <= 20; ++epoch) {
        // Generate synthetic data
        auto x = tensor::randn({10, 2});  // 10 samples, 2 features
        auto y_true = tensor::zeros({10, 1});
        
        // Create target: y = x1^2 + x2^2
        for (size_t i = 0; i < 10; ++i) {
            float x1 = x->data_[i * 2];
            float x2 = x->data_[i * 2 + 1];
            y_true->data_[i] = x1 * x1 + x2 * x2;
        }
        
        // Training step
        optimizer.zero_grad();
        auto y_pred = model->forward(x);
        auto loss = F::mse_loss(y_pred, y_true);
        loss->backward();
        optimizer.step();
        
        if (epoch % 5 == 0) {
            std::cout << "Epoch " << std::setw(2) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(6) 
                      << loss->data_[0] << std::endl;
        }
    }
    
    std::cout << "Training completed!" << std::endl;
}

void demo_performance_features() {
    std::cout << "\n⚡ Performance Features Demo" << std::endl;
    std::cout << std::string(40, '=') << std::endl;
    
    // Configuration
    config::print_config();
    
    // Performance timing
    {
        NEURALFORGE_TIMER("Matrix Multiplication");
        auto a = tensor::randn({1000, 1000});
        auto b = tensor::randn({1000, 1000});
        auto c = a->matmul(b);
        std::cout << "Computed 1000x1000 matrix multiplication" << std::endl;
    }
    
    // Memory usage
    auto memory_usage = utils::memory::get_memory_usage_mb();
    if (memory_usage > 0) {
        std::cout << "Current memory usage: " << std::fixed << std::setprecision(1) 
                  << memory_usage << " MB" << std::endl;
    }
    
    // SIMD demonstration
    std::cout << "\nSIMD support: " << (config::is_simd_available() ? "Available" : "Not available") << std::endl;
    
    // Thread configuration
    std::cout << "Using " << config::get_num_threads() << " threads for parallel operations" << std::endl;
}

void demo_version_info() {
    std::cout << "\n📋 Version Information" << std::endl;
    std::cout << std::string(40, '=') << std::endl;
    
    std::cout << Version::get_build_info() << std::endl;
}

int main() {
    try {
        std::cout << "🔥 NeuralForge - Simple Demo" << std::endl;
        std::cout << "High-Performance Neural Networks in C++" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        // Enable performance optimizations
        config::set_simd_enabled(true);
        config::set_num_threads(std::min(4, static_cast<int>(std::thread::hardware_concurrency())));
        
        // Run demos
        demo_version_info();
        demo_tensor_operations();
        demo_automatic_differentiation();
        demo_neural_network();
        demo_training_loop();
        demo_performance_features();
        
        std::cout << "\n🎉 Demo completed successfully!" << std::endl;
        std::cout << "\nKey features demonstrated:" << std::endl;
        std::cout << "✅ High-performance tensor operations" << std::endl;
        std::cout << "✅ Automatic differentiation" << std::endl;
        std::cout << "✅ Neural network layers" << std::endl;
        std::cout << "✅ Training with optimizers" << std::endl;
        std::cout << "✅ SIMD optimizations" << std::endl;
        std::cout << "✅ Multi-threading support" << std::endl;
        
        std::cout << "\nFor more examples, see:" << std::endl;
        std::cout << "- mnist_classification: Full MNIST training example" << std::endl;
        std::cout << "- regression_example: Advanced regression demo" << std::endl;
        std::cout << "- benchmarks/: Performance benchmarking suite" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Demo failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 