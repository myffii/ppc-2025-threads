#include "stl/moiseev_a_mult_mat/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool moiseev_a_mult_mat_stl::MultMatSTL::PreProcessingImpl() {
  unsigned int input_size_a = task_data->inputs_count[0];
  unsigned int input_size_b = task_data->inputs_count[1];

  auto *in_ptr_a = reinterpret_cast<double *>(task_data->inputs[0]);
  auto *in_ptr_b = reinterpret_cast<double *>(task_data->inputs[1]);

  matrix_a_ = std::vector<double>(in_ptr_a, in_ptr_a + input_size_a);
  matrix_b_ = std::vector<double>(in_ptr_b, in_ptr_b + input_size_b);

  unsigned int output_size = task_data->outputs_count[0];
  matrix_c_ = std::vector<double>(output_size, 0.0);

  matrix_size_ = static_cast<int>(std::sqrt(input_size_a));
  block_size_ = static_cast<int>(std::sqrt(matrix_size_));
  if (matrix_size_ % block_size_ != 0) {
    block_size_ = 1;
  }
  num_blocks_ = matrix_size_ / block_size_;

  return true;
}

bool moiseev_a_mult_mat_stl::MultMatSTL::ValidationImpl() {
  return (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
         (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

void moiseev_a_mult_mat_stl::MultMatSTL::MultiplyBlock(const BlockDesc &desc) {
  int block_row = desc.row;
  int block_col = desc.col;
  int block_step = desc.step;

  int a_block_col = (block_row + block_step) % num_blocks_;
  int b_block_row = a_block_col;

  int row_base = block_row * block_size_;
  int col_base = block_col * block_size_;
  int a_col_base = a_block_col * block_size_;
  int b_row_base = b_block_row * block_size_;

  for (int i = 0; i < block_size_; ++i) {
    for (int j = 0; j < block_size_; ++j) {
      double sum = 0.0;
      for (int k = 0; k < block_size_; ++k) {
        double a = matrix_a_[((row_base + i) * matrix_size_) + (a_col_base + k)];
        double b = matrix_b_[((b_row_base + k) * matrix_size_) + (col_base + j)];
        sum += a * b;
      }
      matrix_c_[((row_base + i) * matrix_size_) + (col_base + j)] += sum;
    }
  }
}

bool moiseev_a_mult_mat_stl::MultMatSTL::RunImpl() {
  const std::size_t total_block_rows = num_blocks_;
  const std::size_t num_threads = std::min<std::size_t>(ppc::util::GetPPCNumThreads(), total_block_rows);
  std::vector<std::thread> threads(num_threads);

  auto worker = [&](std::size_t thread_index) {
    std::size_t base = total_block_rows / num_threads;
    std::size_t extra = total_block_rows % num_threads;
    std::size_t start = (thread_index * base) + std::min(thread_index, extra);
    std::size_t count = base + (thread_index < extra ? 1 : 0);
    std::size_t end = start + count;

    for (int br = static_cast<int>(start); br < static_cast<int>(end); ++br) {
      for (int bc = 0; bc < num_blocks_; ++bc) {
        for (int bs = 0; bs < num_blocks_; ++bs) {
          MultiplyBlock({.row = br, .col = bc, .step = bs});
        }
      }
    }
  };

  for (std::size_t t = 0; t < num_threads; ++t) {
    threads[t] = std::thread(worker, t);
  }
  for (auto &th : threads) {
    th.join();
  }
  return true;
}

bool moiseev_a_mult_mat_stl::MultMatSTL::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);
  std::ranges::copy(matrix_c_, out_ptr);
  return true;
}