#include "omp/gromov_a_fox_algorithm/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

bool gromov_a_fox_algorithm_omp::TestTaskOpenMP::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  if (input_size % 2 != 0) {
    return false;
  }

  unsigned int matrix_size = input_size / 2;
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);

  A_ = std::vector<double>(in_ptr, in_ptr + matrix_size);
  B_ = std::vector<double>(in_ptr + matrix_size, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0.0);

  n_ = static_cast<int>(std::sqrt(matrix_size));
  if (n_ * n_ != static_cast<int>(matrix_size)) {
    return false;
  }

  block_size_ = n_ / 2;
  for (int i = 1; i <= n_; ++i) {
    if (n_ % i == 0) {
      block_size_ = i;
      break;
    }
  }
  return block_size_ > 0;
}

bool gromov_a_fox_algorithm_omp::TestTaskOpenMP::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  if (input_size % 2 != 0) {
    return false;
  }
  unsigned int matrix_size = input_size / 2;
  auto sqrt_matrix_size = static_cast<unsigned int>(std::sqrt(matrix_size));
  return matrix_size == task_data->outputs_count[0] && sqrt_matrix_size * sqrt_matrix_size == matrix_size;
}

bool gromov_a_fox_algorithm_omp::TestTaskOpenMP::RunImpl() {
  int num_blocks = (n_ + block_size_ - 1) / block_size_;

  std::vector<double>& a_ref = A_;
  std::vector<double>& b_ref = B_;
  std::vector<double>& output_ref = output_;
  int n_ref = n_;
  int block_size_ref = block_size_;

  for (int stage = 0; stage < num_blocks; ++stage) {
#pragma omp parallel for default(none) shared(a_ref, b_ref, output_ref, n_ref, block_size_ref, stage)
    for (int i = 0; i < n_ref; i += block_size_ref) {
      for (int j = 0; j < n_ref; j += block_size_ref) {
        for (int bi = i; bi < i + block_size_ref && bi < n_ref; ++bi) {
          for (int bj = j; bj < j + block_size_ref && bj < n_ref; ++bj) {
            int start_k = stage * block_size_ref;
            for (int bk = start_k; bk < std::min((stage + 1) * block_size_ref, n_ref); ++bk) {
              output_ref[(bi * n_ref) + bj] += a_ref[(bi * n_ref) + bk] * b_ref[(bk * n_ref) + bj];
            }
          }
        }
      }
    }
  }
  return true;
}

bool gromov_a_fox_algorithm_omp::TestTaskOpenMP::PostProcessingImpl() {
  std::ranges::copy(output_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}