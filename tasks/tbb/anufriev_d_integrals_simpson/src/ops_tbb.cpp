#include "tbb/anufriev_d_integrals_simpson/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <vector>

namespace {

int SimpsonCoeff(int i, int n) {
  if (i == 0 || i == n) {
    return 1;
  }
  if (i % 2 != 0) {
    return 4;
  }
  return 2;
}
}  // namespace

namespace anufriev_d_integrals_simpson_tbb {

double IntegralsSimpsonTBB::FunctionN(const std::vector<double>& coords) const {
  switch (func_code_) {
    case 0: {
      double s = 0.0;
      for (double c : coords) {
        s += c * c;
      }
      return s;
    }
    case 1: {
      double val = 1.0;
      for (size_t i = 0; i < coords.size(); i++) {
        if (i % 2 == 0) {
          val *= std::sin(coords[i]);
        } else {
          val *= std::cos(coords[i]);
        }
      }
      return val;
    }
    default:
      return 0.0;
  }
}

bool IntegralsSimpsonTBB::PreProcessingImpl() {
  if (task_data->inputs.empty() || task_data->inputs[0] == nullptr) {
    return false;
  }

  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  size_t in_size_bytes = task_data->inputs_count[0];
  size_t num_doubles = in_size_bytes / sizeof(double);

  if (num_doubles < 1) {
    return false;
  }

  int d = static_cast<int>(in_ptr[0]);
  if (d <= 0) {
    return false;
  }

  size_t required_elements = 1 + static_cast<size_t>(3 * d) + 1;
  if (num_doubles < required_elements) {
    return false;
  }

  dimension_ = d;
  a_.resize(dimension_);
  b_.resize(dimension_);
  n_.resize(dimension_);

  int idx_ptr = 1;
  for (int i = 0; i < dimension_; i++) {
    a_[i] = in_ptr[idx_ptr++];
    b_[i] = in_ptr[idx_ptr++];
    double n_double = in_ptr[idx_ptr++];
    if (std::floor(n_double) != n_double || n_double > static_cast<double>(std::numeric_limits<int>::max())) {
      return false;
    }
    n_[i] = static_cast<int>(n_double);

    if (n_[i] <= 0 || (n_[i] % 2) != 0) {
      return false;
    }
    if (a_[i] > b_[i]) {
    }
  }

  double func_code_double = in_ptr[idx_ptr];
  if (std::floor(func_code_double) != func_code_double ||
      func_code_double > static_cast<double>(std::numeric_limits<int>::max()) ||
      func_code_double < static_cast<double>(std::numeric_limits<int>::min())) {
    return false;
  }
  func_code_ = static_cast<int>(func_code_double);

  result_ = 0.0;

  return true;
}

bool IntegralsSimpsonTBB::ValidationImpl() {
  if (task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }
  if (task_data->outputs_count.empty() || task_data->outputs_count[0] < sizeof(double)) {
    return false;
  }
  return true;
}

bool IntegralsSimpsonTBB::RunImpl() {
  std::vector<double> steps(dimension_);
  size_t total_points = 1;
  double coeff_mult = 1.0;

  for (int i = 0; i < dimension_; i++) {
    if (n_[i] == 0) {
      return false;
    }
    steps[i] = (b_[i] - a_[i]) / n_[i];
    coeff_mult *= steps[i] / 3.0;
    size_t points_in_dim = static_cast<size_t>(n_[i]) + 1;
    if (total_points > std::numeric_limits<size_t>::max() / points_in_dim) {
      return false;
    }
    total_points *= points_in_dim;
  }

  double total_sum = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, total_points), 0.0,
      [&](const tbb::blocked_range<size_t>& r, double running_sum) {
        std::vector<double> coords(dimension_);
        std::vector<int> current_idx(dimension_);

        for (size_t k = r.begin(); k != r.end(); ++k) {
          double current_coeff_prod = 1.0;
          size_t current_k = k;

          for (int dim = 0; dim < dimension_; ++dim) {
            size_t points_in_this_dim = static_cast<size_t>(n_[dim]) + 1;
            size_t index_in_this_dim = current_k % points_in_this_dim;
            current_idx[dim] = static_cast<int>(index_in_this_dim);
            current_k /= points_in_this_dim;

            coords[dim] = a_[dim] + current_idx[dim] * steps[dim];
            current_coeff_prod *= SimpsonCoeff(current_idx[dim], n_[dim]);
          }

          running_sum += current_coeff_prod * FunctionN(coords);
        }
        return running_sum;
      },
      [](double x, double y) { return x + y; });

  result_ = coeff_mult * total_sum;
  return true;
}

bool IntegralsSimpsonTBB::PostProcessingImpl() {
  try {
    auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
    out_ptr[0] = result_;
  } catch (const std::exception& e) {
    std::cerr << "Error during PostProcessing: " << e.what() << '\n';
    return false;
  }
  return true;
}

}  // namespace anufriev_d_integrals_simpson_tbb
