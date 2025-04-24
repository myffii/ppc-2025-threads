#include "omp/poroshin_v_multi_integral_with_trapez_method/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

void poroshin_v_multi_integral_with_trapez_method_omp::TestTaskOpenMP::CountMultiIntegralTrapezMethodOmp() {
  const int dimensions = static_cast<int>(limits_.size());
  std::vector<double> h(dimensions);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < dimensions; ++i) {
    h[i] = (limits_[i].second - limits_[i].first) / n_[i];
  }

  std::vector<std::vector<double>> weights(dimensions);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < dimensions; ++i) {
    weights[i].resize(n_[i] + 1);
    for (int j = 0; j <= n_[i]; ++j) {
      weights[i][j] = (j == 0 || j == n_[i]) ? 0.5 : 1.0;
    }
  }

  double integral = 0.0;

#pragma omp parallel reduction(+ : integral)
  {
    std::vector<double> vars(dimensions);
    std::vector<int> indices(dimensions, 0);

    int total_points = 1;
    for (int n : n_) {
      total_points *= (n + 1);
    }

#pragma omp for schedule(static)
    for (int linear_idx = 0; linear_idx < total_points; ++linear_idx) {
      int idx = linear_idx;
      for (int dim = dimensions - 1; dim >= 0; --dim) {
        indices[dim] = idx % (n_[dim] + 1);
        idx /= (n_[dim] + 1);
      }

      double weight = 1.0;
      for (int dim = 0; dim < dimensions; ++dim) {
        vars[dim] = limits_[dim].first + indices[dim] * h[dim];
        weight *= weights[dim][indices[dim]];
      }

      integral += func_(vars) * weight;
    }
  }

  double volume = 1.0;
#pragma omp parallel for reduction(* : volume)
  for (int i = 0; i < dimensions; ++i) {
    volume *= h[i];
  }

  res_ = integral * volume;
}

bool poroshin_v_multi_integral_with_trapez_method_omp::TestTaskOpenMP::PreProcessingImpl() {
  n_.resize(dim_);
  limits_.resize(dim_);
  for (size_t i = 0; i < dim_; i++) {
    n_[i] = reinterpret_cast<int *>(task_data->inputs[0])[i];
    limits_[i].first = reinterpret_cast<double *>(task_data->inputs[1])[i];
    limits_[i].second = reinterpret_cast<double *>(task_data->inputs[2])[i];
  }
  res_ = 0;
  return true;
}

bool poroshin_v_multi_integral_with_trapez_method_omp::TestTaskOpenMP::ValidationImpl() {
  return (task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == 1 && task_data->inputs_count[0] == dim_);
}

bool poroshin_v_multi_integral_with_trapez_method_omp::TestTaskOpenMP::RunImpl() {
  CountMultiIntegralTrapezMethodOmp();
  return true;
}

bool poroshin_v_multi_integral_with_trapez_method_omp::TestTaskOpenMP::PostProcessingImpl() {
  reinterpret_cast<double *>(task_data->outputs[0])[0] = res_;
  return true;
}