#include "stl/poroshin_v_multi_integral_with_trapez_method/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

void poroshin_v_multi_integral_with_trapez_method_stl::TestTaskSTL::CalculateData(
    std::vector<double>& h, std::vector<std::vector<double>>& weights, int& total_points, const int& dimensions) {
  for (int i = 0; i < dimensions; ++i) {
    h[i] = (limits_[i].second - limits_[i].first) / n_[i];

    total_points *= (n_[i] + 1);

    weights[i].resize(n_[i] + 1);
    for (int j = 0; j <= n_[i]; ++j) {
      weights[i][j] = (j == 0 || j == n_[i]) ? 0.5 : 1.0;
    }
  }
}

void poroshin_v_multi_integral_with_trapez_method_stl::TestTaskSTL::CountMultiIntegralTrapezMethodStl() {
  const int dimensions = static_cast<int>(limits_.size());
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<double> h(dimensions);
  std::vector<std::vector<double>> weights(dimensions);
  int total_points = 1;
  CalculateData(h, weights, total_points, dimensions);
  std::vector<std::pair<double, double>> thread_results(num_threads, {0.0, 0.0});
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  const int chunk_size = (total_points + num_threads - 1) / num_threads;

  for (int i = 0; i < num_threads; ++i) {
    int start = i * chunk_size;
    int end = std::min(start + chunk_size, total_points);

    if (start < end) {
      threads.emplace_back([this, dimensions, start, end, &h, &weights, total_points, &thread_results, i]() {
        double local_integral = 0.0;
        std::vector<double> vars(dimensions);
        std::vector<int> indices(dimensions, 0);

        for (int linear_idx = start; linear_idx < end; ++linear_idx) {
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

        double chunk_fraction = static_cast<double>(end - start) / total_points;
        double local_volume = chunk_fraction;
        for (double hi : h) {
          local_volume *= hi;
        }

        thread_results[i] = {local_integral, local_volume};
      });
    }
  }

  for (auto& thread : threads) {
    thread.join();
  }

  double integral = 0.0;
  double volume = 0.0;
  for (const auto& result : thread_results) {
    integral += result.first;
    volume += result.second;
  }

  res_ = integral * volume;
}

bool poroshin_v_multi_integral_with_trapez_method_stl::TestTaskSTL::PreProcessingImpl() {
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

bool poroshin_v_multi_integral_with_trapez_method_stl::TestTaskSTL::ValidationImpl() {
  return (task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == 1 && task_data->inputs_count[0] == dim_);
}

bool poroshin_v_multi_integral_with_trapez_method_stl::TestTaskSTL::RunImpl() {
  CountMultiIntegralTrapezMethodStl();
  return true;
}

bool poroshin_v_multi_integral_with_trapez_method_stl::TestTaskSTL::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = res_;
  return true;
}
