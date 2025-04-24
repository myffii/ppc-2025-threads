#include "tbb/poroshin_v_multi_integral_with_trapez_method/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

void poroshin_v_multi_integral_with_trapez_method_tbb::TestTaskTBB::CountMultiIntegralTrapezMethodTbb() {
  const int dimensions = static_cast<int>(limits_.size());
  std::vector<double> h(dimensions);

  tbb::parallel_for(tbb::blocked_range<int>(0, dimensions), [&](const tbb::blocked_range<int>& range) {
    for (int i = range.begin(); i != range.end(); ++i) {
      h[i] = (limits_[i].second - limits_[i].first) / n_[i];
    }
  });

  std::vector<std::vector<double>> weights(dimensions);
  tbb::parallel_for(tbb::blocked_range<int>(0, dimensions), [&](const tbb::blocked_range<int>& range) {
    for (int i = range.begin(); i != range.end(); ++i) {
      weights[i].resize(n_[i] + 1);
      for (int j = 0; j <= n_[i]; ++j) {
        weights[i][j] = (j == 0 || j == n_[i]) ? 0.5 : 1.0;
      }
    }
  });

  double integral = tbb::parallel_reduce(
      tbb::blocked_range<int>(0,
                              [&]() {
                                int total_points = 1;
                                for (int n : n_) {
                                  total_points *= (n + 1);
                                }
                                return total_points;
                              }()),
      0.0,
      [&](const tbb::blocked_range<int>& range, double local_integral) {
        std::vector<double> vars(dimensions);
        std::vector<int> indices(dimensions, 0);

        for (int linear_idx = range.begin(); linear_idx != range.end(); ++linear_idx) {
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

          local_integral += func_(vars) * weight;
        }
        return local_integral;
      },
      std::plus<>());

  double volume = tbb::parallel_reduce(
      tbb::blocked_range<int>(0, dimensions), 1.0,
      [&](const tbb::blocked_range<int>& range, double local_volume) {
        for (int i = range.begin(); i != range.end(); ++i) {
          local_volume *= h[i];
        }
        return local_volume;
      },
      std::multiplies<>());

  res_ = integral * volume;
}

bool poroshin_v_multi_integral_with_trapez_method_tbb::TestTaskTBB::PreProcessingImpl() {
  n_.resize(dim_);
  limits_.resize(dim_);
  for (size_t i = 0; i < dim_; i++) {
    n_[i] = reinterpret_cast<int*>(task_data->inputs[0])[i];
    limits_[i].first = reinterpret_cast<double*>(task_data->inputs[1])[i];
    limits_[i].second = reinterpret_cast<double*>(task_data->inputs[2])[i];
  }
  res_ = 0;
  return true;
}

bool poroshin_v_multi_integral_with_trapez_method_tbb::TestTaskTBB::ValidationImpl() {
  return (task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == 1 && task_data->inputs_count[0] == dim_);
}

bool poroshin_v_multi_integral_with_trapez_method_tbb::TestTaskTBB::RunImpl() {
  CountMultiIntegralTrapezMethodTbb();
  return true;
}

bool poroshin_v_multi_integral_with_trapez_method_tbb::TestTaskTBB::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = res_;
  return true;
}