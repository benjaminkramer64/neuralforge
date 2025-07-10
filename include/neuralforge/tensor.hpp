/**
 * @file tensor.hpp
 * @brief Minimal Cross-Platform Tensor Implementation
 * 
 * NeuralForge - Advanced Neural Network Library for C++
 */

#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <iomanip>
#include <stdexcept>

namespace neuralforge {

// Forward declarations
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

/**
 * @brief Shape type for tensor dimensions
 */
using Shape = std::vector<size_t>;

/**
 * @brief Minimal Tensor class for demonstration
 */
class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    // Core data
    std::vector<float> data_;
    Shape shape_;
    std::vector<size_t> strides_;
    
    // Automatic differentiation
    bool requires_grad_;
    TensorPtr grad_;
    
public:
    // Default constructor
    Tensor() : requires_grad_(false) {}
    
    // Shape constructor (zero-initialized)
    explicit Tensor(const Shape& shape, bool requires_grad = false)
        : shape_(shape), requires_grad_(requires_grad) {
        size_t total_size = compute_size(shape);
        data_.resize(total_size, 0.0f);
        compute_strides();
        
        if (requires_grad_) {
            grad_ = std::make_shared<Tensor>(shape, false);
        }
    }
    
    // Data constructor
    Tensor(const Shape& shape, const std::vector<float>& data, bool requires_grad = false)
        : shape_(shape), data_(data), requires_grad_(requires_grad) {
        if (data.size() != compute_size(shape)) {
            throw std::invalid_argument("Data size doesn't match shape");
        }
        compute_strides();
        
        if (requires_grad_) {
            grad_ = std::make_shared<Tensor>(shape, false);
        }
    }
    
    /**
     * @brief Factory methods
     */
    static TensorPtr zeros(const Shape& shape, bool requires_grad = false) {
        return std::make_shared<Tensor>(shape, requires_grad);
    }
    
    static TensorPtr ones(const Shape& shape, bool requires_grad = false) {
        auto tensor = std::make_shared<Tensor>(shape, requires_grad);
        std::fill(tensor->data_.begin(), tensor->data_.end(), 1.0f);
        return tensor;
    }
    
    static TensorPtr randn(const Shape& shape, bool requires_grad = false, 
                          float mean = 0.0f, float std = 1.0f) {
        auto tensor = std::make_shared<Tensor>(shape, requires_grad);
        
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        std::normal_distribution<float> dist(mean, std);
        
        for (auto& val : tensor->data_) {
            val = dist(gen);
        }
        
        return tensor;
    }
    
    /**
     * @brief Basic properties
     */
    size_t size() const { return data_.size(); }
    size_t ndim() const { return shape_.size(); }
    const Shape& shape() const { return shape_; }
    
    // Linear indexing
    float& operator[](size_t index) { return data_[index]; }
    const float& operator[](size_t index) const { return data_[index]; }
    
    /**
     * @brief Operations
     */
    TensorPtr add(const TensorPtr& other) const {
        if (shape_ != other->shape_) {
            throw std::runtime_error("Shape mismatch for addition");
        }
        
        auto result = std::make_shared<Tensor>(shape_, requires_grad_ || other->requires_grad_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result->data_[i] = data_[i] + other->data_[i];
        }
        return result;
    }
    
    TensorPtr sub(const TensorPtr& other) const {
        if (shape_ != other->shape_) {
            throw std::runtime_error("Shape mismatch for subtraction");
        }
        
        auto result = std::make_shared<Tensor>(shape_, requires_grad_ || other->requires_grad_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result->data_[i] = data_[i] - other->data_[i];
        }
        return result;
    }
    
    TensorPtr mul(const TensorPtr& other) const {
        if (shape_ != other->shape_) {
            throw std::runtime_error("Shape mismatch for multiplication");
        }
        
        auto result = std::make_shared<Tensor>(shape_, requires_grad_ || other->requires_grad_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result->data_[i] = data_[i] * other->data_[i];
        }
        return result;
    }
    
    TensorPtr mul(float scalar) const {
        auto result = std::make_shared<Tensor>(shape_, requires_grad_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result->data_[i] = data_[i] * scalar;
        }
        return result;
    }
    
    TensorPtr add(float scalar) const {
        auto result = std::make_shared<Tensor>(shape_, requires_grad_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result->data_[i] = data_[i] + scalar;
        }
        return result;
    }
    
    TensorPtr relu() const {
        auto result = std::make_shared<Tensor>(shape_, requires_grad_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result->data_[i] = std::max(0.0f, data_[i]);
        }
        return result;
    }
    
    TensorPtr tanh() const {
        auto result = std::make_shared<Tensor>(shape_, requires_grad_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result->data_[i] = std::tanh(data_[i]);
        }
        return result;
    }
    
    TensorPtr sigmoid() const {
        auto result = std::make_shared<Tensor>(shape_, requires_grad_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result->data_[i] = 1.0f / (1.0f + std::exp(-data_[i]));
        }
        return result;
    }
    
    TensorPtr matmul(const TensorPtr& other) const {
        if (shape_.size() != 2 || other->shape_.size() != 2) {
            throw std::runtime_error("Matrix multiplication requires 2D tensors");
        }
        
        if (shape_[1] != other->shape_[0]) {
            throw std::runtime_error("Incompatible shapes for matrix multiplication");
        }
        
        Shape result_shape = {shape_[0], other->shape_[1]};
        auto result = std::make_shared<Tensor>(result_shape, requires_grad_ || other->requires_grad_);
        
        // Basic matrix multiplication
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < other->shape_[1]; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < shape_[1]; ++k) {
                    sum += data_[i * shape_[1] + k] * other->data_[k * other->shape_[1] + j];
                }
                result->data_[i * other->shape_[1] + j] = sum;
            }
        }
        
        return result;
    }
    
    TensorPtr reshape(const Shape& new_shape) const {
        if (compute_size(new_shape) != size()) {
            throw std::invalid_argument("New shape incompatible with data size");
        }
        
        auto result = std::make_shared<Tensor>();
        result->data_ = data_;  // Share data
        result->shape_ = new_shape;
        result->requires_grad_ = requires_grad_;
        result->compute_strides();
        
        return result;
    }
    
    TensorPtr transpose(int dim0 = -2, int dim1 = -1) const {
        if (shape_.size() != 2) {
            throw std::runtime_error("Transpose only implemented for 2D tensors");
        }
        
        Shape new_shape = {shape_[1], shape_[0]};
        auto result = std::make_shared<Tensor>(new_shape, requires_grad_);
        
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                result->data_[j * shape_[0] + i] = data_[i * shape_[1] + j];
            }
        }
        
        return result;
    }
    
    TensorPtr sum() const {
        float total = 0.0f;
        for (const auto& val : data_) {
            total += val;
        }
        
        auto result = std::make_shared<Tensor>(Shape{1}, requires_grad_);
        result->data_[0] = total;
        return result;
    }
    
    TensorPtr mean() const {
        auto sum_tensor = sum();
        return sum_tensor->mul(1.0f / static_cast<float>(size()));
    }
    
    /**
     * @brief Automatic differentiation
     */
    void backward() {
        if (!requires_grad_) {
            throw std::runtime_error("Tensor doesn't require gradients");
        }
        
        if (!grad_) {
            grad_ = std::make_shared<Tensor>(shape_, false);
            if (size() == 1) {
                grad_->data_[0] = 1.0f;
            } else {
                std::fill(grad_->data_.begin(), grad_->data_.end(), 1.0f);
            }
        }
    }
    
    void zero_grad() {
        if (grad_) {
            std::fill(grad_->data_.begin(), grad_->data_.end(), 0.0f);
        }
    }
    
    /**
     * @brief Utilities
     */
    std::string to_string() const {
        std::stringstream ss;
        ss << "Tensor(shape: [";
        for (size_t i = 0; i < shape_.size(); ++i) {
            ss << shape_[i];
            if (i < shape_.size() - 1) ss << ", ";
        }
        ss << "], requires_grad: " << (requires_grad_ ? "true" : "false") << ")\n";
        
        size_t max_print = std::min(size(), size_t(20));
        ss << "[";
        for (size_t i = 0; i < max_print; ++i) {
            ss << std::fixed << std::setprecision(4) << data_[i];
            if (i < max_print - 1) ss << ", ";
        }
        if (max_print < size()) ss << ", ...";
        ss << "]";
        
        return ss.str();
    }

private:
    size_t compute_size(const Shape& shape) const {
        return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    }
    
    void compute_strides() {
        strides_.resize(shape_.size());
        if (!shape_.empty()) {
            strides_[shape_.size() - 1] = 1;
            for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }
        }
    }
};

// Operator overloads
inline TensorPtr operator+(const TensorPtr& a, const TensorPtr& b) { return a->add(b); }
inline TensorPtr operator-(const TensorPtr& a, const TensorPtr& b) { return a->sub(b); }
inline TensorPtr operator*(const TensorPtr& a, const TensorPtr& b) { return a->mul(b); }
inline TensorPtr operator+(const TensorPtr& a, float b) { return a->add(b); }
inline TensorPtr operator*(const TensorPtr& a, float b) { return a->mul(b); }

inline std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    return os << tensor.to_string();
}

} // namespace neuralforge 