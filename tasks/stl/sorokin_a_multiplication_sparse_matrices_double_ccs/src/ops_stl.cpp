#include "stl/sorokin_a_multiplication_sparse_matrices_double_ccs/include/ops_stl.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace sorokin_a_multiplication_sparse_matrices_double_ccs_stl {

void ProcessFirstPass(int j, int m, const std::vector<int>& a_col_ptr, const std::vector<int>& a_row_indices,
                      const std::vector<int>& b_col_ptr, const std::vector<int>& b_row_indices,
                      std::vector<int>& col_sizes) {
  std::vector<char> temp_used(m, 0);
  for (int t = b_col_ptr[j]; t < b_col_ptr[j + 1]; ++t) {
    const int row_b = b_row_indices[t];
    for (int i = a_col_ptr[row_b]; i < a_col_ptr[row_b + 1]; ++i) {
      const int row_a = a_row_indices[i];
      if (temp_used[row_a] == 0) {
        temp_used[row_a] = 1;
        col_sizes[j]++;
      }
    }
  }
}

void ProcessSecondPass(int j, int m, const std::vector<int>& a_col_ptr, const std::vector<int>& a_row_indices,
                       const std::vector<double>& a_values, const std::vector<int>& b_col_ptr,
                       const std::vector<int>& b_row_indices, const std::vector<double>& b_values,
                       const std::vector<int>& c_col_ptr, std::vector<int>& c_row_indices,
                       std::vector<double>& c_values) {
  std::vector<double> temp_values(m, 0.0);
  std::vector<char> temp_used(m, 0);

  for (int t = b_col_ptr[j]; t < b_col_ptr[j + 1]; ++t) {
    const int row_b = b_row_indices[t];
    const double val_b = b_values[t];
    for (int i = a_col_ptr[row_b]; i < a_col_ptr[row_b + 1]; ++i) {
      const int row_a = a_row_indices[i];
      temp_values[row_a] += a_values[i] * val_b;
      temp_used[row_a] = 1;
    }
  }

  int pos = c_col_ptr[j];
  for (int i = 0; i < m; ++i) {
    if (temp_used[i] != 0) {
      c_row_indices[pos] = i;
      c_values[pos] = temp_values[i];
      pos++;
    }
  }
}

void MultiplyCCS(const std::vector<double>& a_values, const std::vector<int>& a_row_indices, int m,
                 const std::vector<int>& a_col_ptr, const std::vector<double>& b_values,
                 const std::vector<int>& b_row_indices, int k, const std::vector<int>& b_col_ptr,
                 std::vector<double>& c_values, std::vector<int>& c_row_indices, int n, std::vector<int>& c_col_ptr) {
  c_col_ptr.resize(n + 1);
  c_col_ptr[0] = 0;
  std::vector<int> col_sizes(n, 0);

  int num_threads = ppc::util::GetPPCNumThreads();
  num_threads = std::min(num_threads, n);

  std::vector<std::thread> threads;
  std::atomic<int> next_j(0);

  auto first_pass_worker = [&]() {
    int j = 0;
    while ((j = next_j.fetch_add(1, std::memory_order_relaxed)) < n) {
      ProcessFirstPass(j, m, a_col_ptr, a_row_indices, b_col_ptr, b_row_indices, col_sizes);
    }
  };

  threads.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(first_pass_worker);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  for (int j = 0; j < n; ++j) {
    c_col_ptr[j + 1] = c_col_ptr[j] + col_sizes[j];
  }

  const int total_non_zero = c_col_ptr[n];
  c_values.resize(total_non_zero);
  c_row_indices.resize(total_non_zero);

  threads.clear();
  next_j.store(0);

  auto second_pass_worker = [&]() {
    int j = 0;
    while ((j = next_j.fetch_add(1, std::memory_order_relaxed)) < n) {
      ProcessSecondPass(j, m, a_col_ptr, a_row_indices, a_values, b_col_ptr, b_row_indices, b_values, c_col_ptr,
                        c_row_indices, c_values);
    }
  };

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(second_pass_worker);
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

}  // namespace sorokin_a_multiplication_sparse_matrices_double_ccs_stl

bool sorokin_a_multiplication_sparse_matrices_double_ccs_stl::TestTaskTBB::PreProcessingImpl() {
  // Init value for input and output
  M_ = static_cast<int>(task_data->inputs_count[0]);
  K_ = static_cast<int>(task_data->inputs_count[1]);
  N_ = static_cast<int>(task_data->inputs_count[2]);
  auto* current_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  A_values_ = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[3]);
  current_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
  std::vector<double> a_row_indices_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[4]);
  A_row_indices_.resize(a_row_indices_d.size());
  std::ranges::transform(a_row_indices_d.begin(), a_row_indices_d.end(), A_row_indices_.begin(),
                         [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
  std::vector<double> a_col_ptr_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[5]);
  A_col_ptr_.resize(a_col_ptr_d.size());
  std::ranges::transform(a_col_ptr_d.begin(), a_col_ptr_d.end(), A_col_ptr_.begin(),
                         [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double*>(task_data->inputs[3]);
  B_values_ = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[6]);
  current_ptr = reinterpret_cast<double*>(task_data->inputs[4]);
  std::vector<double> b_row_indices_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[7]);
  B_row_indices_.resize(b_row_indices_d.size());
  std::ranges::transform(b_row_indices_d.begin(), b_row_indices_d.end(), B_row_indices_.begin(),
                         [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double*>(task_data->inputs[5]);
  std::vector<double> b_col_ptr_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[8]);
  B_col_ptr_.resize(b_col_ptr_d.size());
  std::ranges::transform(b_col_ptr_d.begin(), b_col_ptr_d.end(), B_col_ptr_.begin(),
                         [](double x) { return static_cast<int>(x); });
  return true;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_stl::TestTaskTBB::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 && task_data->inputs_count[2] > 0;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_stl::TestTaskTBB::RunImpl() {
  MultiplyCCS(A_values_, A_row_indices_, M_, A_col_ptr_, B_values_, B_row_indices_, K_, B_col_ptr_, C_values_,
              C_row_indices_, N_, C_col_ptr_);
  return true;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_stl::TestTaskTBB::PostProcessingImpl() {
  std::vector<double> c_row_indices_d(C_row_indices_.size());
  std::vector<double> c_col_ptr_d(C_col_ptr_.size());
  std::ranges::transform(C_row_indices_.begin(), C_row_indices_.end(), c_row_indices_d.begin(),
                         [](int x) { return static_cast<double>(x); });
  std::ranges::transform(C_col_ptr_.begin(), C_col_ptr_.end(), c_col_ptr_d.begin(),
                         [](int x) { return static_cast<double>(x); });
  for (size_t i = 0; i < C_values_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = C_values_[i];
  }
  for (size_t i = 0; i < c_row_indices_d.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[1])[i] = c_row_indices_d[i];
  }
  for (size_t i = 0; i < c_col_ptr_d.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[2])[i] = c_col_ptr_d[i];
  }
  return true;
}
