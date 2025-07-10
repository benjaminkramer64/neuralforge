/**
 * @file nn.hpp
 * @brief Neural Network Modules and Layers
 * 
 * NeuralForge - Advanced Neural Network Library for C++
 * 
 * This module provides:
 * - Modular neural network layers (Linear, Conv2D, etc.)
 * - Activation functions (ReLU, Tanh, Sigmoid, etc.)
 * - Loss functions (MSE, CrossEntropy, etc.)
 * - Optimizers (SGD, Adam, AdamW, etc.)
 * - Training utilities and model management
 */

#pragma once

#include "tensor.hpp"
#include <vector>
#include <memory>
#include <random>
#include <unordered_map>
#include <string>

namespace neuralforge::nn {

/**
 * @brief Base class for all neural network modules
 */
class Module {
public:
    Module() = default;
    virtual ~Module() = default;
    
    // Pure virtual forward pass
    virtual TensorPtr forward(TensorPtr input) = 0;
    
    // Get all parameters for optimization
    virtual std::vector<TensorPtr> parameters() = 0;
    
    // Set training mode
    virtual void train(bool mode = true) { training_ = mode; }
    virtual void eval() { train(false); }
    
    // Get module name
    virtual std::string name() const = 0;
    
    // Utility for parameter initialization
    void xavier_uniform_(TensorPtr tensor, float gain = 1.0f) {
        float fan_in = tensor->shape()[0];
        float fan_out = tensor->shape().size() > 1 ? tensor->shape()[1] : 1;
        float std = gain * std::sqrt(2.0f / (fan_in + fan_out));
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-std, std);
        
        for (auto& val : tensor->data_) {
            val = dist(gen);
        }
    }
    
    void kaiming_uniform_(TensorPtr tensor, float a = 0.0f) {
        float fan_in = tensor->shape()[0];
        float gain = std::sqrt(2.0f / (1.0f + a * a));
        float std = gain / std::sqrt(fan_in);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-std, std);
        
        for (auto& val : tensor->data_) {
            val = dist(gen);
        }
    }

protected:
    bool training_ = true;
};

/**
 * @brief Linear (fully connected) layer
 */
class Linear : public Module {
private:
    TensorPtr weight_;
    TensorPtr bias_;
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;

public:
    Linear(size_t in_features, size_t out_features, bool bias = true)
        : in_features_(in_features), out_features_(out_features), use_bias_(bias) {
        
        // Initialize weight and bias tensors
        weight_ = std::make_shared<Tensor>(Shape{out_features, in_features}, true);
        
        if (use_bias_) {
            bias_ = std::make_shared<Tensor>(Shape{out_features}, true);
        }
        
        reset_parameters();
    }
    
    void reset_parameters() {
        kaiming_uniform_(weight_, std::sqrt(5.0f));
        
        if (use_bias_) {
            float bound = 1.0f / std::sqrt(in_features_);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(-bound, bound);
            
            for (auto& val : bias_->data_) {
                val = dist(gen);
            }
        }
    }
    
    TensorPtr forward(TensorPtr input) override {
        // input: (batch_size, in_features) or (in_features,)
        // weight: (out_features, in_features)
        // output: (batch_size, out_features) or (out_features,)
        
        TensorPtr output;
        
        if (input->shape().size() == 1) {
            // Single sample: input is (in_features,)
            if (input->shape()[0] != in_features_) {
                throw std::runtime_error("Input feature size mismatch");
            }
            
            // output = weight @ input
            output = weight_->matmul(input->reshape({in_features_, 1}));
            output = output->reshape({out_features_});
            
        } else if (input->shape().size() == 2) {
            // Batch: input is (batch_size, in_features)
            if (input->shape()[1] != in_features_) {
                throw std::runtime_error("Input feature size mismatch");
            }
            
            // output = input @ weight^T
            auto weight_t = weight_->transpose();
            output = input->matmul(weight_t);
            
        } else {
            throw std::runtime_error("Linear layer expects 1D or 2D input");
        }
        
        // Add bias if enabled
        if (use_bias_) {
            output = output->add(bias_);
        }
        
        return output;
    }
    
    std::vector<TensorPtr> parameters() override {
        std::vector<TensorPtr> params = {weight_};
        if (use_bias_) {
            params.push_back(bias_);
        }
        return params;
    }
    
    std::string name() const override {
        return "Linear(" + std::to_string(in_features_) + ", " + 
               std::to_string(out_features_) + ")";
    }
    
    // Getters
    TensorPtr weight() const { return weight_; }
    TensorPtr bias() const { return bias_; }
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
};

/**
 * @brief Activation Functions
 */

class ReLU : public Module {
public:
    TensorPtr forward(TensorPtr input) override {
        return input->relu();
    }
    
    std::vector<TensorPtr> parameters() override {
        return {};  // No parameters
    }
    
    std::string name() const override {
        return "ReLU()";
    }
};

class Tanh : public Module {
public:
    TensorPtr forward(TensorPtr input) override {
        return input->tanh();
    }
    
    std::vector<TensorPtr> parameters() override {
        return {};
    }
    
    std::string name() const override {
        return "Tanh()";
    }
};

class Sigmoid : public Module {
public:
    TensorPtr forward(TensorPtr input) override {
        return input->sigmoid();
    }
    
    std::vector<TensorPtr> parameters() override {
        return {};
    }
    
    std::string name() const override {
        return "Sigmoid()";
    }
};

class LeakyReLU : public Module {
private:
    float negative_slope_;

public:
    LeakyReLU(float negative_slope = 0.01f) : negative_slope_(negative_slope) {}
    
    TensorPtr forward(TensorPtr input) override {
        // LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
        auto positive = input->relu();
        auto negative = input->mul(negative_slope_)->relu()->mul(-1.0f)->add(input->mul(negative_slope_));
        return positive->add(negative);
    }
    
    std::vector<TensorPtr> parameters() override {
        return {};
    }
    
    std::string name() const override {
        return "LeakyReLU(" + std::to_string(negative_slope_) + ")";
    }
};

class Softmax : public Module {
private:
    int dim_;

public:
    Softmax(int dim = -1) : dim_(dim) {}
    
    TensorPtr forward(TensorPtr input) override {
        // Softmax with numerical stability
        // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        
        if (input->shape().size() == 1) {
            // 1D case
            float max_val = *std::max_element(input->data_.begin(), input->data_.end());
            auto shifted = input->add(-max_val);
            auto exp_vals = shifted->exp();
            auto sum_exp = exp_vals->sum();
            return exp_vals->div(sum_exp);
            
        } else if (input->shape().size() == 2) {
            // 2D case (batch processing)
            auto result = std::make_shared<Tensor>(input->shape(), input->requires_grad_);
            
            size_t batch_size = input->shape()[0];
            size_t num_classes = input->shape()[1];
            
            for (size_t b = 0; b < batch_size; ++b) {
                // Find max for numerical stability
                float max_val = input->data_[b * num_classes];
                for (size_t c = 1; c < num_classes; ++c) {
                    max_val = std::max(max_val, input->data_[b * num_classes + c]);
                }
                
                // Compute softmax
                float sum_exp = 0.0f;
                for (size_t c = 0; c < num_classes; ++c) {
                    float exp_val = std::exp(input->data_[b * num_classes + c] - max_val);
                    result->data_[b * num_classes + c] = exp_val;
                    sum_exp += exp_val;
                }
                
                // Normalize
                for (size_t c = 0; c < num_classes; ++c) {
                    result->data_[b * num_classes + c] /= sum_exp;
                }
            }
            
            return result;
        } else {
            throw std::runtime_error("Softmax only supports 1D and 2D tensors");
        }
    }
    
    std::vector<TensorPtr> parameters() override {
        return {};
    }
    
    std::string name() const override {
        return "Softmax(dim=" + std::to_string(dim_) + ")";
    }
};

/**
 * @brief Sequential container for chaining modules
 */
class Sequential : public Module {
private:
    std::vector<std::shared_ptr<Module>> modules_;

public:
    template<typename... Modules>
    Sequential(Modules... modules) {
        (modules_.push_back(std::make_shared<typename std::decay<Modules>::type>(modules)), ...);
    }
    
    void add_module(std::shared_ptr<Module> module) {
        modules_.push_back(module);
    }
    
    TensorPtr forward(TensorPtr input) override {
        TensorPtr output = input;
        for (auto& module : modules_) {
            output = module->forward(output);
        }
        return output;
    }
    
    std::vector<TensorPtr> parameters() override {
        std::vector<TensorPtr> all_params;
        for (auto& module : modules_) {
            auto params = module->parameters();
            all_params.insert(all_params.end(), params.begin(), params.end());
        }
        return all_params;
    }
    
    void train(bool mode = true) override {
        Module::train(mode);
        for (auto& module : modules_) {
            module->train(mode);
        }
    }
    
    std::string name() const override {
        std::string result = "Sequential(\n";
        for (size_t i = 0; i < modules_.size(); ++i) {
            result += "  (" + std::to_string(i) + "): " + modules_[i]->name() + "\n";
        }
        result += ")";
        return result;
    }
    
    // Access modules by index
    std::shared_ptr<Module>& operator[](size_t index) {
        return modules_[index];
    }
    
    size_t size() const { return modules_.size(); }
};

/**
 * @brief Loss Functions
 */

class MSELoss {
public:
    static TensorPtr forward(TensorPtr input, TensorPtr target) {
        // MSE = mean((input - target)^2)
        auto diff = input->sub(target);
        auto squared = diff->mul(diff);
        return squared->mean();
    }
};

class CrossEntropyLoss {
public:
    static TensorPtr forward(TensorPtr input, TensorPtr target) {
        // CrossEntropy = -sum(target * log(softmax(input)))
        // input: (batch_size, num_classes) - logits
        // target: (batch_size,) - class indices or (batch_size, num_classes) - one-hot
        
        Softmax softmax(-1);
        auto probs = softmax.forward(input);
        
        if (target->shape().size() == 1) {
            // Target is class indices
            auto result = std::make_shared<Tensor>(Shape{1}, input->requires_grad_);
            float loss = 0.0f;
            
            size_t batch_size = input->shape()[0];
            size_t num_classes = input->shape()[1];
            
            for (size_t b = 0; b < batch_size; ++b) {
                int target_class = static_cast<int>(target->data_[b]);
                float prob = probs->data_[b * num_classes + target_class];
                loss -= std::log(std::max(prob, 1e-7f));  // Avoid log(0)
            }
            
            result->data_[0] = loss / batch_size;
            return result;
            
        } else {
            // Target is one-hot encoded
            auto log_probs = probs->log();
            auto product = target->mul(log_probs);
            auto sum = product->sum();
            return sum->mul(-1.0f / input->shape()[0]);
        }
    }
};

class BCELoss {
public:
    static TensorPtr forward(TensorPtr input, TensorPtr target) {
        // BCE = -mean(target * log(input) + (1 - target) * log(1 - input))
        auto log_input = input->log();
        auto one_minus_target = target->mul(-1.0f)->add(1.0f);
        auto one_minus_input = input->mul(-1.0f)->add(1.0f);
        auto log_one_minus_input = one_minus_input->log();
        
        auto loss = target->mul(log_input)->add(one_minus_target->mul(log_one_minus_input));
        return loss->mean()->mul(-1.0f);
    }
};

/**
 * @brief Optimizers
 */

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    
protected:
    std::vector<TensorPtr> parameters_;
    
public:
    Optimizer(const std::vector<TensorPtr>& parameters) : parameters_(parameters) {}
};

class SGD : public Optimizer {
private:
    float lr_;
    float momentum_;
    float weight_decay_;
    std::vector<TensorPtr> velocity_;

public:
    SGD(const std::vector<TensorPtr>& parameters, float lr = 0.01f, 
        float momentum = 0.0f, float weight_decay = 0.0f)
        : Optimizer(parameters), lr_(lr), momentum_(momentum), weight_decay_(weight_decay) {
        
        // Initialize velocity buffers for momentum
        if (momentum_ > 0.0f) {
            for (const auto& param : parameters_) {
                velocity_.push_back(Tensor::zeros(param->shape()));
            }
        }
    }
    
    void step() override {
        for (size_t i = 0; i < parameters_.size(); ++i) {
            auto& param = parameters_[i];
            if (!param->grad_) continue;
            
            auto grad = param->grad_;
            
            // Apply weight decay
            if (weight_decay_ > 0.0f) {
                grad = grad->add(param->mul(weight_decay_));
            }
            
            // Apply momentum
            if (momentum_ > 0.0f) {
                // v = momentum * v + grad
                velocity_[i] = velocity_[i]->mul(momentum_)->add(grad);
                grad = velocity_[i];
            }
            
            // Update parameters: param = param - lr * grad
            for (size_t j = 0; j < param->size(); ++j) {
                param->data_[j] -= lr_ * grad->data_[j];
            }
        }
    }
    
    void zero_grad() override {
        for (auto& param : parameters_) {
            param->zero_grad();
        }
    }
};

class Adam : public Optimizer {
private:
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    int step_count_;
    
    std::vector<TensorPtr> m_;  // First moment
    std::vector<TensorPtr> v_;  // Second moment

public:
    Adam(const std::vector<TensorPtr>& parameters, float lr = 0.001f,
         float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
         float weight_decay = 0.0f)
        : Optimizer(parameters), lr_(lr), beta1_(beta1), beta2_(beta2), 
          eps_(eps), weight_decay_(weight_decay), step_count_(0) {
        
        // Initialize moment buffers
        for (const auto& param : parameters_) {
            m_.push_back(Tensor::zeros(param->shape()));
            v_.push_back(Tensor::zeros(param->shape()));
        }
    }
    
    void step() override {
        step_count_++;
        
        for (size_t i = 0; i < parameters_.size(); ++i) {
            auto& param = parameters_[i];
            if (!param->grad_) continue;
            
            auto grad = param->grad_;
            
            // Apply weight decay
            if (weight_decay_ > 0.0f) {
                grad = grad->add(param->mul(weight_decay_));
            }
            
            // Update biased first moment estimate
            // m = beta1 * m + (1 - beta1) * grad
            m_[i] = m_[i]->mul(beta1_)->add(grad->mul(1.0f - beta1_));
            
            // Update biased second raw moment estimate
            // v = beta2 * v + (1 - beta2) * grad^2
            auto grad_squared = grad->mul(grad);
            v_[i] = v_[i]->mul(beta2_)->add(grad_squared->mul(1.0f - beta2_));
            
            // Compute bias-corrected first moment estimate
            float m_hat_scale = 1.0f / (1.0f - std::pow(beta1_, step_count_));
            auto m_hat = m_[i]->mul(m_hat_scale);
            
            // Compute bias-corrected second raw moment estimate
            float v_hat_scale = 1.0f / (1.0f - std::pow(beta2_, step_count_));
            auto v_hat = v_[i]->mul(v_hat_scale);
            
            // Update parameters
            for (size_t j = 0; j < param->size(); ++j) {
                float denominator = std::sqrt(v_hat->data_[j]) + eps_;
                param->data_[j] -= lr_ * m_hat->data_[j] / denominator;
            }
        }
    }
    
    void zero_grad() override {
        for (auto& param : parameters_) {
            param->zero_grad();
        }
    }
};

/**
 * @brief Training utilities
 */

class Trainer {
private:
    std::shared_ptr<Module> model_;
    std::shared_ptr<Optimizer> optimizer_;
    
public:
    Trainer(std::shared_ptr<Module> model, std::shared_ptr<Optimizer> optimizer)
        : model_(model), optimizer_(optimizer) {}
    
    float train_step(TensorPtr input, TensorPtr target) {
        model_->train(true);
        optimizer_->zero_grad();
        
        auto output = model_->forward(input);
        auto loss = CrossEntropyLoss::forward(output, target);
        
        loss->backward();
        optimizer_->step();
        
        return loss->data_[0];
    }
    
    float eval_step(TensorPtr input, TensorPtr target) {
        model_->eval();
        
        auto output = model_->forward(input);
        auto loss = CrossEntropyLoss::forward(output, target);
        
        return loss->data_[0];
    }
    
    void train_epoch(const std::vector<std::pair<TensorPtr, TensorPtr>>& dataset) {
        float total_loss = 0.0f;
        
        for (const auto& [input, target] : dataset) {
            float loss = train_step(input, target);
            total_loss += loss;
        }
        
        std::cout << "Average loss: " << total_loss / dataset.size() << std::endl;
    }
};

} // namespace neuralforge::nn 