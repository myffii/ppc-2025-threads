#include "tbb/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "oneapi/tbb/parallel_for.h"
void lysov_i_matrix_multiplication_fox_algorithm_tbb::ProcessBlock(const std::vector<double> &a,
                                                                   const std::vector<double> &b, std::vector<double> &c,
                                                                   std::size_t i, std::size_t j,
                                                                   std::size_t a_block_row, std::size_t block_size,
                                                                   std::size_t n) {
  std::size_t block_h = std::min(block_size, n - (i * block_size));
  std::size_t block_w = std::min(block_size, n - (j * block_size));
  std::size_t block_k = std::min(block_size, n - (a_block_row * block_size));

  double *c_ptr = &c[((i * block_size) * n) + (j * block_size)];
  const double *a_ptr = &a[((i * block_size) * n) + (a_block_row * block_size)];
  const double *b_ptr = &b[((a_block_row * block_size) * n) + (j * block_size)];

  for (std::size_t ii = 0; ii < block_h; ++ii) {
    for (std::size_t jj = 0; jj < block_w; ++jj) {
      double sum = 0.0;
      const double *a_row = a_ptr + (ii * n);
      const double *b_col = b_ptr + (jj);

      for (std::size_t kk = 0; kk + 1 < block_k; kk += 2) {
        sum += a_row[kk] * b_col[kk * n] + a_row[kk + 1] * b_col[(kk + 1) * n];
      }
      if (block_k % 2 != 0) {
        sum += a_row[block_k - 1] * b_col[(block_k - 1) * n];
      }

      c_ptr[(ii * n) + jj] += sum;
    }
  }
}

bool lysov_i_matrix_multiplication_fox_algorithm_tbb::TestTaskTBB::PreProcessingImpl() {
  // Init value for input and output
  n_ = reinterpret_cast<std::size_t *>(task_data->inputs[0])[0];
  block_size_ = reinterpret_cast<std::size_t *>(task_data->inputs[3])[0];
  a_.resize(n_ * n_);
  b_.resize(n_ * n_);
  c_.clear();
  c_.resize(n_ * n_, 0.0);
  std::copy(reinterpret_cast<double *>(task_data->inputs[1]),
            reinterpret_cast<double *>(task_data->inputs[1]) + (n_ * n_), a_.begin());
  std::copy(reinterpret_cast<double *>(task_data->inputs[2]),
            reinterpret_cast<double *>(task_data->inputs[2]) + (n_ * n_), b_.begin());
  return true;
}

bool lysov_i_matrix_multiplication_fox_algorithm_tbb::TestTaskTBB::ValidationImpl() {
  n_ = reinterpret_cast<std::size_t *>(task_data->inputs[0])[0];
  block_size_ = reinterpret_cast<std::size_t *>(task_data->inputs[3])[0];
  if (task_data->inputs_count.size() != 3 || task_data->outputs_count.size() != 1) {
    return false;
  }
  if (task_data->inputs_count[1] != n_ * n_ || task_data->inputs_count[0] != n_ * n_) {
    return false;
  }
  return task_data->outputs_count[0] == n_ * n_ && block_size_ > 0;
}

bool lysov_i_matrix_multiplication_fox_algorithm_tbb::TestTaskTBB::RunImpl() {
  const int num_blocks = static_cast<int>((n_ + block_size_ - 1) / block_size_);
  tbb::parallel_for(0, num_blocks * num_blocks, [&](int index) {
    int i = index / num_blocks;
    int j = index % num_blocks;
    for (int step = 0; step < num_blocks; ++step) {
      int a_block_row = (i + step) % num_blocks;
      ProcessBlock(a_, b_, c_, i, j, a_block_row, block_size_, n_);
    }
  });

  return true;
}

bool lysov_i_matrix_multiplication_fox_algorithm_tbb::TestTaskTBB::PostProcessingImpl() {
  std::ranges::copy(c_, reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
