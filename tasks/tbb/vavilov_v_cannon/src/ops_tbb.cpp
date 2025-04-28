#include "tbb/vavilov_v_cannon/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/parallel_for.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/task_arena.h"

bool vavilov_v_cannon_tbb::CannonTBB::PreProcessingImpl() {
  N_ = static_cast<int>(std::sqrt(task_data->inputs_count[0]));
  num_blocks_ = static_cast<int>(task_data->inputs_count[2]);
  block_size_ = N_ / num_blocks_;

  auto* a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* b = reinterpret_cast<double*>(task_data->inputs[1]);
  A_.assign(a, a + (N_ * N_));
  B_.assign(b, b + (N_ * N_));
  C_.assign(N_ * N_, 0);

  return true;
}

bool vavilov_v_cannon_tbb::CannonTBB::ValidationImpl() {
  if (task_data->inputs_count[0] != task_data->inputs_count[1] ||
      task_data->outputs_count[0] != task_data->inputs_count[0]) {
    return false;
  }

  auto n = static_cast<int>(std::sqrt(task_data->inputs_count[0]));
  auto num_blocks = static_cast<int>(task_data->inputs_count[2]);
  return n % num_blocks == 0;
}

void vavilov_v_cannon_tbb::CannonTBB::InitialShift() {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;
  tbb::parallel_for(tbb::blocked_range2d<int>(0, num_blocks_, 0, num_blocks_), [&](const tbb::blocked_range2d<int>& r) {
    for (int bi = r.rows().begin(); bi != r.rows().end(); ++bi) {
      for (int bj = r.cols().begin(); bj != r.cols().end(); ++bj) {
        int src_row = (bi + bj) % num_blocks_;
        int src_col = (bj + bi) % num_blocks_;
        for (int i = 0; i < block_size_; ++i) {
          for (int j = 0; j < block_size_; ++j) {
            B_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
                b_tmp[(((src_row * block_size_) + i) * N_) + ((bj * block_size_) + j)];
            A_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
                a_tmp[(((bi * block_size_) + i) * N_) + ((src_col * block_size_) + j)];
          }
        }
      }
    }
  });
}

void vavilov_v_cannon_tbb::CannonTBB::CopyBlocksToLocal(std::vector<double>& a_block, std::vector<double>& b_block,
                                                        int base_row, int base_col) {
  for (int i = 0; i < block_size_ && base_row + i < N_; ++i) {
    for (int k = 0; k < block_size_ && base_col + k < N_; ++k) {
      a_block[(i * block_size_) + k] = A_[((base_row + i) * N_) + (base_col + k)];
      b_block[(k * block_size_) + i] = B_[((base_row + k) * N_) + (base_col + i)];
    }
  }
}

void vavilov_v_cannon_tbb::CannonTBB::ComputeBlock(const std::vector<double>& a_block,
                                                   const std::vector<double>& b_block, int base_row, int base_col) {
  for (int i = 0; i < block_size_ && base_row + i < N_; ++i) {
    int row = base_row + i;
    for (int j = 0; j < block_size_ && base_col + j < N_; ++j) {
      int col = base_col + j;
      double temp = 0.0;
      int k = 0;

      for (; k <= block_size_ - 4; k += 4) {
        temp += a_block[(i * block_size_) + k] * b_block[(k * block_size_) + j] +
                a_block[(i * block_size_) + k + 1] * b_block[((k + 1) * block_size_) + j] +
                a_block[(i * block_size_) + k + 2] * b_block[((k + 2) * block_size_) + j] +
                a_block[(i * block_size_) + k + 3] * b_block[((k + 3) * block_size_) + j];
      }

      for (; k < block_size_ && base_row + k < N_; ++k) {
        temp += a_block[(i * block_size_) + k] * b_block[(k * block_size_) + j];
      }

      C_[(row * N_) + col] += temp;
    }
  }
}

void vavilov_v_cannon_tbb::CannonTBB::BlockMultiply() {
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range2d<int>(0, num_blocks_, 0, num_blocks_),
      [&](const oneapi::tbb::blocked_range2d<int>& r) {
        std::vector<double> a_block(block_size_ * block_size_);
        std::vector<double> b_block(block_size_ * block_size_);

        for (int bi = r.rows().begin(); bi != r.rows().end(); ++bi) {
          for (int bj = r.cols().begin(); bj != r.cols().end(); ++bj) {
            int base_row = bi * block_size_;
            int base_col = bj * block_size_;

            CopyBlocksToLocal(a_block, b_block, base_row, base_col);
            ComputeBlock(a_block, b_block, base_row, base_col);
          }
        }
      },
      oneapi::tbb::auto_partitioner());
}

void vavilov_v_cannon_tbb::CannonTBB::ShiftBlocks() {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;
  tbb::parallel_for(tbb::blocked_range2d<int>(0, num_blocks_, 0, num_blocks_), [&](const tbb::blocked_range2d<int>& r) {
    for (int bi = r.rows().begin(); bi != r.rows().end(); ++bi) {
      for (int bj = r.cols().begin(); bj != r.cols().end(); ++bj) {
        int src_row = (bi + 1) % num_blocks_;
        int src_col = (bj + 1) % num_blocks_;
        for (int i = 0; i < block_size_; ++i) {
          for (int j = 0; j < block_size_; ++j) {
            B_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
                b_tmp[(((src_row * block_size_) + i) * N_) + ((bj * block_size_) + j)];
            A_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
                a_tmp[(((bi * block_size_) + i) * N_) + ((src_col * block_size_) + j)];
          }
        }
      }
    }
  });
}

bool vavilov_v_cannon_tbb::CannonTBB::RunImpl() {
  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  arena.execute([&]() {
    InitialShift();
    for (int iter = 0; iter < num_blocks_; ++iter) {
      BlockMultiply();
      ShiftBlocks();
    }
  });
  return true;
}

bool vavilov_v_cannon_tbb::CannonTBB::PostProcessingImpl() {
  std::ranges::copy(C_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
