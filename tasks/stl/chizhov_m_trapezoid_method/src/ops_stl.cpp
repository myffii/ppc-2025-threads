#include "stl/chizhov_m_trapezoid_method/include/ops_stl.hpp"

#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

double chizhov_m_trapezoid_method_stl::ComputeWeight(const std::vector<double>& point,
                                                     const std::vector<double>& lower_limits,
                                                     const std::vector<double>& upper_limits) {
  double weight = 1.0;
  for (size_t j = 0; j < point.size(); j++) {
    if (point[j] == lower_limits[j] || point[j] == upper_limits[j]) {
      weight *= 1.0;
    } else {
      weight *= 2.0;
    }
  }
  return weight;
}

double chizhov_m_trapezoid_method_stl::TrapezoidMethod(Function& f, size_t div, size_t dim,
                                                       std::vector<double>& lower_limits,
                                                       std::vector<double>& upper_limits) {
  int int_dim = static_cast<int>(dim);
  std::vector<double> h(int_dim);
  std::vector<int> steps(int_dim);

  for (int i = 0; i < int_dim; i++) {
    steps[i] = static_cast<int>(div);
    h[i] = (upper_limits[i] - lower_limits[i]) / steps[i];
  }

  long long total_nodes = 1;
  for (const auto& step : steps) {
    total_nodes *= (step + 1);
  }

  double result = 0.0;
  auto num_threads = ppc::util::GetPPCNumThreads();
  std::vector<double> local_results(num_threads, 0.0);
  std::vector<std::thread> threads(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads[t] = std::thread([&, t]() {
      long long start = t * (total_nodes / num_threads);
      long long end = (t + 1) * (total_nodes / num_threads);
      if (t == num_threads - 1) {
        end = total_nodes;
      }

      for (long long i = start; i < end; ++i) {
        int temp = static_cast<int>(i);
        std::vector<double> point(int_dim);

        for (int j = 0; j < int_dim; j++) {
          int node_index = temp % (steps[j] + 1);
          point[j] = lower_limits[j] + node_index * h[j];
          temp /= (steps[j] + 1);
        }

        double weight = ComputeWeight(point, lower_limits, upper_limits);

        local_results[t] += weight * f(point);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  for (const auto& local_result : local_results) {
    result += local_result;
  }

  for (int i = 0; i < int_dim; i++) {
    result *= h[i] / 2.0;
  }

  return std::round(result * 100.0) / 100.0;
}

bool chizhov_m_trapezoid_method_stl::TestTaskSTL::PreProcessingImpl() {
  int* divisions_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  div_ = *divisions_ptr;

  int* dimension_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
  dim_ = *dimension_ptr;

  auto* limit_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
  for (int i = 0; i < static_cast<int>(task_data->inputs_count[2]); i += 2) {
    lower_limits_.push_back(limit_ptr[i]);
    upper_limits_.push_back(limit_ptr[i + 1]);
  }
  auto* ptr_f = reinterpret_cast<std::function<double(const std::vector<double>&)>*>(task_data->inputs[3]);
  f_ = *ptr_f;

  return true;
}

bool chizhov_m_trapezoid_method_stl::TestTaskSTL::ValidationImpl() {
  int* divisions_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  int* dimension_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
  if (*divisions_ptr <= 0 || *dimension_ptr <= 0) {
    return false;
  }
  if (task_data->inputs_count[2] % 2 != 0) {
    return false;
  }
  auto* limit_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
  for (int i = 0; i < static_cast<int>(task_data->inputs_count[2]); i += 2) {
    if (limit_ptr[i] >= limit_ptr[i + 1]) {
      return false;
    }
  }

  return true;
}

bool chizhov_m_trapezoid_method_stl::TestTaskSTL::RunImpl() {
  res_ = TrapezoidMethod(f_, div_, dim_, lower_limits_, upper_limits_);

  return true;
}

bool chizhov_m_trapezoid_method_stl::TestTaskSTL::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = res_;

  return true;
}