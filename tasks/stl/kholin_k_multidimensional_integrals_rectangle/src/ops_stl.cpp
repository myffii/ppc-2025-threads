#include "stl/kholin_k_multidimensional_integrals_rectangle/include/ops_stl.hpp"

#include <cstddef>
#include <functional>
#include <numeric>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

double kholin_k_multidimensional_integrals_rectangle_stl::TestTaskSTL::Integrate(
    const Function& f, const std::vector<double>& l_limits, const std::vector<double>& u_limits,
    const std::vector<double>& h, std::vector<double> f_values, int curr_index_dim, size_t dim, double n) {
  if (curr_index_dim == static_cast<int>(dim)) {
    return f(f_values);
  }

  const int total_steps = static_cast<int>(n);
  double sum = 0.0;

  if (curr_index_dim == 0) {
    const int num_threads = ppc::util::GetPPCNumThreads();
    std::vector<std::thread> threads(num_threads);
    std::vector<double> thread_sums(num_threads, 0.0);

    const int base_steps = total_steps / num_threads;
    const int remainder = total_steps % num_threads;

    for (int t = 0; t < num_threads; ++t) {
      const int start = (t * base_steps) + (t < remainder ? t : remainder);
      const int end = start + base_steps + (t < remainder ? 1 : 0);

      threads[t] = std::thread([&, start, end, t]() {
        std::vector<double> local_f_values = f_values;
        double local_sum = 0.0;
        for (int i = start; i < end; ++i) {
          local_f_values[0] = l_limits[0] + (i + 0.5) * h[0];
          local_sum += Integrate(f, l_limits, u_limits, h, local_f_values, 1, dim, n);
        }
        thread_sums[t] = local_sum;
      });
    }

    for (auto& th : threads) {
      if (th.joinable()) {
        th.join();
      }
    }
    sum = std::accumulate(thread_sums.begin(), thread_sums.end(), 0.0) * h[0];
  } else {
    for (int i = 0; i < total_steps; ++i) {
      f_values[curr_index_dim] = l_limits[curr_index_dim] + (i + 0.5) * h[curr_index_dim];
      sum += Integrate(f, l_limits, u_limits, h, f_values, curr_index_dim + 1, dim, n);
    }
    sum *= h[curr_index_dim];
  }

  return sum;
}

double kholin_k_multidimensional_integrals_rectangle_stl::TestTaskSTL::IntegrateWithRectangleMethod(
    const Function& f, std::vector<double>& f_values, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, size_t dim, double n, std::vector<double> h) {
  for (size_t i = 0; i < dim; ++i) {
    h[i] = (u_limits[i] - l_limits[i]) / n;
  }

  return Integrate(f, l_limits, u_limits, h, f_values, 0, dim, n);
}

double kholin_k_multidimensional_integrals_rectangle_stl::TestTaskSTL::RunMultistepSchemeMethodRectangle(
    const Function& f, std::vector<double> f_values, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, size_t dim, double n) {
  std::vector<double> h(dim);
  double i_n = 0.0;

  i_n = IntegrateWithRectangleMethod(f, f_values, l_limits, u_limits, dim, n, h);
  return i_n;
}

bool kholin_k_multidimensional_integrals_rectangle_stl::TestTaskSTL::PreProcessingImpl() {
  // Init value for input and output
  sz_values_ = task_data->inputs_count[0];
  sz_lower_limits_ = task_data->inputs_count[1];
  sz_upper_limits_ = task_data->inputs_count[2];

  auto* ptr_dim = reinterpret_cast<size_t*>(task_data->inputs[0]);
  dim_ = *ptr_dim;

  auto* ptr_f_values = reinterpret_cast<double*>(task_data->inputs[1]);
  f_values_.assign(ptr_f_values, ptr_f_values + sz_values_);

  auto* ptr_f = reinterpret_cast<std::function<double(const std::vector<double>&)>*>(task_data->inputs[2]);
  f_ = *ptr_f;

  auto* ptr_lower_limits = reinterpret_cast<double*>(task_data->inputs[3]);
  lower_limits_.assign(ptr_lower_limits, ptr_lower_limits + sz_lower_limits_);

  auto* ptr_upper_limits = reinterpret_cast<double*>(task_data->inputs[4]);
  upper_limits_.assign(ptr_upper_limits, ptr_upper_limits + sz_upper_limits_);

  auto* ptr_start_n = reinterpret_cast<double*>(task_data->inputs[5]);
  start_n_ = *ptr_start_n;

  result_ = 0.0;
  count_level_par_ = 1;
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_stl::TestTaskSTL::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[1] > 0U && task_data->inputs_count[2] > 0U;
}

bool kholin_k_multidimensional_integrals_rectangle_stl::TestTaskSTL::RunImpl() {
  result_ = RunMultistepSchemeMethodRectangle(f_, f_values_, lower_limits_, upper_limits_, dim_, start_n_);
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_stl::TestTaskSTL::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}