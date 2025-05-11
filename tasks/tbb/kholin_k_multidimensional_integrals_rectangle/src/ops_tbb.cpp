#include "tbb/kholin_k_multidimensional_integrals_rectangle/include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/partitioner.h>
#include <tbb/tbb.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

double kholin_k_multidimensional_integrals_rectangle_tbb::TestTaskTBB::Integrate(
    const Function& f, const std::vector<double>& l_limits, const std::vector<double>& u_limits,
    const std::vector<double>& h, std::vector<double>& f_values, int curr_index_dim, size_t dim, double n) {
  if (curr_index_dim == static_cast<int>(dim)) {
    return f(f_values);
  }

  const double step = h[curr_index_dim];
  const double start_pos = l_limits[curr_index_dim] + (0.5 * step);

  return tbb::parallel_reduce(
             tbb::blocked_range<int>(0, static_cast<int>(n), 1000), 0.0,
             [&](const tbb::blocked_range<int>& r, double local_sum) {
               std::vector<double> local_f_values = f_values;
               for (int i = r.begin(); i < r.end(); ++i) {
                 local_f_values[curr_index_dim] = start_pos + static_cast<double>(i) * step;
                 local_sum += Integrate(f, l_limits, u_limits, h, local_f_values, curr_index_dim + 1, dim, n);
               }
               return local_sum;
             },
             std::plus<>(), tbb::auto_partitioner()) *
         step;
}

double kholin_k_multidimensional_integrals_rectangle_tbb::TestTaskTBB::IntegrateWithRectangleMethod(
    const Function& f, std::vector<double>& f_values, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, size_t dim, double n, std::vector<double>& h) {
  for (size_t i = 0; i < dim; ++i) {
    h[i] = (u_limits[i] - l_limits[i]) / n;
  }
  return kholin_k_multidimensional_integrals_rectangle_tbb::TestTaskTBB::Integrate(f, l_limits, u_limits, h, f_values,
                                                                                   0, dim, n);
}

double kholin_k_multidimensional_integrals_rectangle_tbb::TestTaskTBB::RunMultistepSchemeMethodRectangle(
    const Function& f, std::vector<double> f_values, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, size_t dim, double n) {
  std::vector<double> h(dim);
  return kholin_k_multidimensional_integrals_rectangle_tbb::TestTaskTBB::IntegrateWithRectangleMethod(
      f, f_values, l_limits, u_limits, dim, n, h);
}

bool kholin_k_multidimensional_integrals_rectangle_tbb::TestTaskTBB::PreProcessingImpl() {
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
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_tbb::TestTaskTBB::ValidationImpl() {
  return task_data->inputs_count[1] > 0U && task_data->inputs_count[2] > 0U;
}

bool kholin_k_multidimensional_integrals_rectangle_tbb::TestTaskTBB::RunImpl() {
  result_ = kholin_k_multidimensional_integrals_rectangle_tbb::TestTaskTBB::RunMultistepSchemeMethodRectangle(
      f_, f_values_, lower_limits_, upper_limits_, dim_, start_n_);
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_tbb::TestTaskTBB::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}