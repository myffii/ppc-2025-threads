#include "tbb/gromov_a_fox_algorithm/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "oneapi/tbb/parallel_for.h"

namespace {
void FoxBlockMul(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c, int n,
                 int block_size, int stage, int i, int j, int k) {
  for (int bi = i; bi < std::min(i + block_size, n); ++bi) {
    for (int bj = j; bj < std::min(j + block_size, n); ++bj) {
      double sum = 0.0;
      for (int bk = k; bk < std::min(k + block_size, n); ++bk) {
        sum += a[(bi * n) + bk] * b[(bk * n) + bj];
      }
      c[(bi * n) + bj] += sum;
    }
  }
}
}  // namespace

bool gromov_a_fox_algorithm_tbb::TestTaskTBB::PreProcessingImpl() {
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

  block_size_ = static_cast<int>(std::sqrt(n_));
  for (int i = block_size_; i >= 1; --i) {
    if (n_ % i == 0) {
      block_size_ = i;
      break;
    }
  }
  for (int i = block_size_ + 1; i <= n_; ++i) {
    if (n_ % i == 0) {
      if (std::abs(i - static_cast<int>(std::sqrt(n_))) < std::abs(block_size_ - static_cast<int>(std::sqrt(n_)))) {
        block_size_ = i;
      }
      break;
    }
  }
  return block_size_ > 0;
}

bool gromov_a_fox_algorithm_tbb::TestTaskTBB::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  if (input_size % 2 != 0) {
    return false;
  }
  unsigned int matrix_size = input_size / 2;
  auto sqrt_matrix_size = static_cast<unsigned int>(std::sqrt(matrix_size));
  return matrix_size == task_data->outputs_count[0] && sqrt_matrix_size * sqrt_matrix_size == matrix_size;
}

bool gromov_a_fox_algorithm_tbb::TestTaskTBB::RunImpl() {
  const int num_blocks = (n_ + block_size_ - 1) / block_size_;

  tbb::parallel_for(0, num_blocks * num_blocks, [&](int index) {
    int i = index / num_blocks;
    int j = index % num_blocks;

    for (int step = 0; step < num_blocks; ++step) {
      int k = (i + step) % num_blocks;
      FoxBlockMul(A_, B_, output_, n_, block_size_, step, i * block_size_, j * block_size_, k * block_size_);
    }
  });

  return true;
}

bool gromov_a_fox_algorithm_tbb::TestTaskTBB::PostProcessingImpl() {
  std::ranges::copy(output_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}