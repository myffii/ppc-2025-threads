#include "omp/anufriev_d_integrals_simpson/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
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

namespace anufriev_d_integrals_simpson_omp {

double IntegralsSimpsonOmp::FunctionN(const std::vector<double>& coords) const {
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

double IntegralsSimpsonOmp::RecursiveSimpsonSum(int dim_index, std::vector<int>& idx,
                                                const std::vector<double>& steps) const {
  if (dim_index == dimension_) {
    double coeff = 1.0;
    std::vector<double> coords(dimension_);
    for (int d = 0; d < dimension_; ++d) {
      coords[d] = a_[d] + idx[d] * steps[d];
      coeff *= SimpsonCoeff(idx[d], n_[d]);
    }
    return coeff * FunctionN(coords);
  }
  double sum = 0.0;
  for (int i = 0; i <= n_[dim_index]; ++i) {
    idx[dim_index] = i;
    sum += RecursiveSimpsonSum(dim_index + 1, idx, steps);
  }
  return sum;
}

bool IntegralsSimpsonOmp::PreProcessingImpl() {
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
  if (d < 1) {
    return false;
  }

  size_t needed_count = 1 + static_cast<size_t>(3 * d) + 1;
  if (num_doubles < needed_count) {
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
    int current_n = static_cast<int>(in_ptr[idx_ptr++]);
    if (current_n <= 0 || (current_n % 2) != 0) {
      return false;
    }
    n_[i] = current_n;
  }

  func_code_ = static_cast<int>(in_ptr[idx_ptr]);

  result_ = 0.0;

  return true;
}

bool IntegralsSimpsonOmp::ValidationImpl() {
  if (task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }
  if (task_data->outputs_count.empty() || task_data->outputs_count[0] < sizeof(double)) {
    return false;
  }
  if (task_data->inputs.empty() || task_data->inputs[0] == nullptr || task_data->inputs_count.empty() ||
      task_data->inputs_count[0] == 0) {
    return false;
  }
  return true;
}

bool IntegralsSimpsonOmp::RunImpl() {
  std::vector<double> steps(dimension_);
  for (int i = 0; i < dimension_; i++) {
    steps[i] = (b_[i] - a_[i]) / n_[i];
  }

  double total_sum = 0.0;

#pragma omp parallel
  {
    std::vector<int> idx(dimension_);

#pragma omp for reduction(+ : total_sum)
    for (int i0 = 0; i0 <= n_[0]; ++i0) {
      idx[0] = i0;

      if (dimension_ == 1) {
        double coeff = SimpsonCoeff(idx[0], n_[0]);
        std::vector<double> coords = {a_[0] + (idx[0] * steps[0])};
        total_sum += coeff * FunctionN(coords);
      } else {
        total_sum += RecursiveSimpsonSum(1, idx, steps);
      }
    }
  }

  double coeff_mult = 1.0;
  for (int i = 0; i < dimension_; i++) {
    coeff_mult *= steps[i] / 3.0;
  }

  result_ = coeff_mult * total_sum;
  return true;
}

bool IntegralsSimpsonOmp::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  out_ptr[0] = result_;
  return true;
}

}  // namespace anufriev_d_integrals_simpson_omp