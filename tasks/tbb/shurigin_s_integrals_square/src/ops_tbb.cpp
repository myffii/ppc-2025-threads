#include "tbb/shurigin_s_integrals_square/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shurigin_s_integrals_square_tbb {

Integral::Integral(std::shared_ptr<ppc::core::TaskData> task_data)
    : Task(task_data),
      down_limits_(1, 0.0),
      up_limits_(1, 0.0),
      counts_(1, 0),
      result_(0.0),
      func_(nullptr),
      dimensions_(1),
      task_data_(std::move(task_data)) {}

void Integral::SetFunction(const std::function<double(double)>& func) {
  func_ = [func](const std::vector<double>& point) {
    if (point.empty()) {
      throw std::runtime_error("Internal error: Point vector is empty in 1D wrapper.");
    }
    return func(point[0]);
  };
  dimensions_ = 1;
  down_limits_.resize(1);
  up_limits_.resize(1);
  counts_.resize(1);
}

void Integral::SetFunction(const std::function<double(const std::vector<double>&)>& func, int dimensions) {
  if (dimensions <= 0) {
    throw std::invalid_argument("Dimensions must be positive.");
  }
  func_ = func;
  dimensions_ = dimensions;
  down_limits_.resize(dimensions_);
  up_limits_.resize(dimensions_);
  counts_.resize(dimensions_);
}

bool Integral::PreProcessingImpl() {
  try {
    if (!task_data_ || task_data_->inputs.empty() || task_data_->inputs[0] == nullptr) {
      throw std::invalid_argument("Invalid input data.");
    }
    auto* inputs = reinterpret_cast<double*>(task_data_->inputs[0]);
    size_t expected_input_size_bytes = static_cast<size_t>(3 * dimensions_) * sizeof(double);
    if (task_data_->inputs_count[0] != expected_input_size_bytes) {
      throw std::invalid_argument("Input data size mismatch. Expected " + std::to_string(expected_input_size_bytes) +
                                  " bytes, got " + std::to_string(task_data_->inputs_count[0]) + " bytes.");
    }

    if (dimensions_ > 1) {
      for (int i = 0; i < dimensions_; ++i) {
        down_limits_[i] = inputs[i];
        up_limits_[i] = inputs[i + dimensions_];
        counts_[i] = static_cast<int>(inputs[i + (2 * dimensions_)]);

        if (counts_[i] <= 0) {
          throw std::invalid_argument("Number of intervals must be positive for all dimensions.");
        }
        if (up_limits_[i] <= down_limits_[i]) {
          throw std::invalid_argument("Upper limit must be greater than lower limit for all dimensions.");
        }
      }
    } else {
      down_limits_[0] = inputs[0];
      up_limits_[0] = inputs[1];
      counts_[0] = static_cast<int>(inputs[2]);

      if (counts_[0] <= 0) {
        throw std::invalid_argument("Number of intervals must be positive.");
      }
      if (up_limits_[0] <= down_limits_[0]) {
        throw std::invalid_argument("Upper limit must be greater than lower limit.");
      }
    }

    result_ = 0.0;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error in PreProcessingImpl: " << e.what() << '\n';
    return false;
  }
}

bool Integral::ValidationImpl() {
  try {
    if (!task_data_) {
      throw std::invalid_argument("task_data is null.");
    }
    if (task_data_->inputs_count.empty() || task_data_->outputs_count.empty()) {
      throw std::invalid_argument("Input or output counts are empty.");
    }

    size_t expected_input_size = 3 * dimensions_ * sizeof(double);
    if (task_data_->inputs_count[0] != expected_input_size) {
      throw std::invalid_argument("Input data size validation failed. Expected " + std::to_string(expected_input_size) +
                                  " bytes, got " + std::to_string(task_data_->inputs_count[0]) + " bytes.");
    }

    if (task_data_->outputs_count[0] != sizeof(double)) {
      throw std::invalid_argument("Output data size validation failed. Expected " + std::to_string(sizeof(double)) +
                                  " bytes, got " + std::to_string(task_data_->outputs_count[0]) + " bytes.");
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error in ValidationImpl: " << e.what() << '\n';
    return false;
  }
}

bool Integral::RunImpl() {
  try {
    if (!func_) {
      throw std::runtime_error("Function is not set.");
    }

    if (dimensions_ > 1) {
      std::vector<double> initial_point(dimensions_);
      result_ = ComputeRecursiveTBB(func_, down_limits_, up_limits_, counts_, dimensions_, initial_point, 0);
      return true;
    }
    return ComputeOneDimensional();

  } catch (const std::exception& e) {
    std::cerr << "Error in RunImpl: " << e.what() << '\n';
    return false;
  }
}

bool Integral::ComputeOneDimensional() {
  const double a = down_limits_[0];
  const double b = up_limits_[0];
  const int n = counts_[0];
  if (n <= 0) {
    std::cerr << "Error: Number of intervals is non-positive in ComputeOneDimensional." << '\n';
    return false;
  }

  const double step = (b - a) / n;
  const double half_step = 0.5 * step;
  const double base = a + half_step;

  double sum = oneapi::tbb::parallel_reduce(
      oneapi::tbb::blocked_range<int>(0, n), 0.0,
      [&](const oneapi::tbb::blocked_range<int>& r, double local_sum) {
        std::vector<double> point(1);
        for (int i = r.begin(); i != r.end(); ++i) {
          point[0] = base + (i * step);
          local_sum += func_(point);
        }
        return local_sum;
      },
      [](double x, double y) { return x + y; });

  result_ = sum * step;
  return true;
}

double Integral::ComputeRecursiveTBB(const std::function<double(const std::vector<double>&)>& f,
                                     const std::vector<double>& a, const std::vector<double>& b,
                                     const std::vector<int>& n, int dim, std::vector<double> point, int current_dim) {
  if (current_dim == dim) {
    return f(point);
  }

  const int current_n = n[current_dim];
  if (current_n <= 0) {
    throw std::runtime_error("Internal error: Non-positive interval count in recursive step.");
  }
  const double step = (b[current_dim] - a[current_dim]) / current_n;
  const double half_step = 0.5 * step;
  const double base = a[current_dim] + half_step;

  double integral_sum = oneapi::tbb::parallel_reduce(
      oneapi::tbb::blocked_range<int>(0, current_n), 0.0,
      [&](const oneapi::tbb::blocked_range<int>& r, double local_sum) {
        std::vector<double> local_point = point;
        for (int i = r.begin(); i != r.end(); ++i) {
          local_point[current_dim] = base + (i * step);
          double recursive_result = ComputeRecursiveTBB(f, a, b, n, dim, local_point, current_dim + 1);
          local_sum += recursive_result;
        }
        return local_sum;
      },
      [](double x, double y) { return x + y; });

  return integral_sum * step;
}

bool Integral::PostProcessingImpl() {
  try {
    if (!task_data_ || task_data_->outputs.empty() || task_data_->outputs[0] == nullptr) {
      throw std::invalid_argument("Invalid output data.");
    }
    auto* outputs = reinterpret_cast<double*>(task_data_->outputs[0]);
    outputs[0] = result_;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error in PostProcessingImpl: " << e.what() << '\n';
    return false;
  }
}

}  // namespace shurigin_s_integrals_square_tbb