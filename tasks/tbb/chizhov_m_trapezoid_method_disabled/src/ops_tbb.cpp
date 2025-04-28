#include "tbb/chizhov_m_trapezoid_method/include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <functional>
#include <vector>

double chizhov_m_trapezoid_method_tbb::TrapezoidMethod(Function& f, size_t div, size_t dim,
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

  const int num_threads = ppc::util::GetPPCNumThreads();
  oneapi::tbb::task_arena arena(num_threads);

  arena.execute([&] {
    result = oneapi::tbb::parallel_reduce(
        tbb::blocked_range<long>(0, total_nodes, 16), 0.0,
        [&](const tbb::blocked_range<long>& r, double local_res) {
          for (long i = r.begin(); i != r.end(); ++i) {
            int temp = static_cast<int>(i);
            double weight = 1.0;
            std::vector<double> point(int_dim);

            for (int j = 0; j < int_dim; j++) {
              int node_index = temp % (steps[j] + 1);
              point[j] = lower_limits[j] + node_index * h[j];
              temp /= (steps[j] + 1);
            }

            for (int j = 0; j < int_dim; j++) {
              if (point[j] == lower_limits[j] || point[j] == upper_limits[j]) {
                weight *= 1.0;
              } else {
                weight *= 2.0;
              }
            }

            local_res += weight * f(point);
          }
          return local_res;
        },
        [](double a, double b) { return a + b; });
  });

  for (int i = 0; i < int_dim; i++) {
    result *= h[i] / 2.0;
  }

  return std::round(result * 100.0) / 100.0;
}

bool chizhov_m_trapezoid_method_tbb::TestTaskTBB::PreProcessingImpl() {
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

bool chizhov_m_trapezoid_method_tbb::TestTaskTBB::ValidationImpl() {
  auto* divisions_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  auto* dimension_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
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

bool chizhov_m_trapezoid_method_tbb::TestTaskTBB::RunImpl() {
  res_ = TrapezoidMethod(f_, div_, dim_, lower_limits_, upper_limits_);
  return true;
}

bool chizhov_m_trapezoid_method_tbb::TestTaskTBB::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = res_;
  return true;
}