#include "stl/vavilov_v_cannon/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool vavilov_v_cannon_stl::CannonSTL::PreProcessingImpl() {
  N_ = static_cast<int>(std::sqrt(task_data->inputs_count[0]));
  num_blocks_ = static_cast<int>(task_data->inputs_count[2]);
  block_size_ = N_ / num_blocks_;

  auto *a = reinterpret_cast<double *>(task_data->inputs[0]);
  auto *b = reinterpret_cast<double *>(task_data->inputs[1]);
  A_.assign(a, a + (N_ * N_));
  B_.assign(b, b + (N_ * N_));
  C_.assign(N_ * N_, 0);

  return true;
}

bool vavilov_v_cannon_stl::CannonSTL::ValidationImpl() {
  if (task_data->inputs_count[0] != task_data->inputs_count[1] ||
      task_data->outputs_count[0] != task_data->inputs_count[0]) {
    return false;
  }

  auto n = static_cast<int>(std::sqrt(task_data->inputs_count[0]));
  auto num_blocks = static_cast<int>(task_data->inputs_count[2]);
  return n % num_blocks == 0;
}

void vavilov_v_cannon_stl::CannonSTL::InitialShift(int num_threads, int blocks_per_thread) {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;
  std::vector<std::thread> threads;

  auto shift_work = [&](int bi_start, int bi_end) {
    for (int bi = bi_start; bi < bi_end; ++bi) {
      for (int bj = 0; bj < num_blocks_; ++bj) {
        int src_row = (bi + bj) % num_blocks_;
        int src_col = (bj + bi) % num_blocks_;
        for (int i = 0; i < block_size_; ++i) {
          for (int j = 0; j < block_size_; ++j) {
            B_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
                b_tmp[(((src_row * block_size_) + i) * N_) + ((bj * block_size_) + j)];
            A_[((bi * block_size_ + i) * N_) + ((bj * block_size_) + j)] =
                a_tmp[(((bi * block_size_) + i) * N_) + ((src_col * block_size_) + j)];
          }
        }
      }
    }
  };

  for (int t = 0; t < num_threads; ++t) {
    int start = t * blocks_per_thread;
    int end = std::min(start + blocks_per_thread, num_blocks_);
    if (start < end) {
      threads.emplace_back(shift_work, start, end);
    }
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void vavilov_v_cannon_stl::CannonSTL::ProcessSingleBlock(int bi, int bj, int bi_start, std::vector<double> &local) {
  for (int i = bi; i < bi + block_size_; i++) {
    for (int j = bj; j < bj + block_size_; j++) {
      double temp = 0.0;
      for (int k = 0; k < block_size_; k++) {
        const int row_a = bi + (i - bi);
        const int col_a = bj + k;
        const int row_b = bi + k;
        const int col_b = bj + (j - bj);
        temp += A_[(row_a * N_) + col_a] * B_[(row_b * N_) + col_b];
      }
      local[((i - bi_start) * N_) + j] += temp;
    }
  }
}

void vavilov_v_cannon_stl::CannonSTL::MergeResults(int num_threads, int bi_range,
                                                   const std::vector<std::vector<double>> &local_c) {
  for (int t = 0; t < num_threads; ++t) {
    const int bi_start = t * bi_range;
    const int bi_end = std::min(bi_start + bi_range, N_);
    if (bi_start < N_) {
      const std::vector<double> &local = local_c[t];
      for (int i = bi_start; i < bi_end; ++i) {
        for (int j = 0; j < N_; ++j) {
          C_[(i * N_) + j] += local[((i - bi_start) * N_) + j];
        }
      }
    }
  }
}

void vavilov_v_cannon_stl::CannonSTL::BlockMultiply(int num_threads, int blocks_per_thread) {
  std::vector<std::vector<double>> local_c(num_threads);
  const int bi_range = blocks_per_thread * block_size_;
  std::vector<std::thread> threads;

  auto process_block_range = [&](int bi_start, int bi_end, int thread_id) {
    std::vector<double> &local = local_c[thread_id];
    local.resize((bi_end - bi_start) * N_, 0.0);

    for (int bi = bi_start; bi < bi_end; bi += block_size_) {
      for (int bj = 0; bj < N_; bj += block_size_) {
        ProcessSingleBlock(bi, bj, bi_start, local);
      }
    }
  };

  for (int t = 0; t < num_threads; ++t) {
    const int bi_start = t * bi_range;
    const int bi_end = std::min(bi_start + bi_range, N_);
    if (bi_start < N_) {
      threads.emplace_back(process_block_range, bi_start, bi_end, t);
    }
  }
  for (auto &thread : threads) {
    thread.join();
  }

  MergeResults(num_threads, bi_range, local_c);
}

void vavilov_v_cannon_stl::CannonSTL::ShiftBlocks(int num_threads, int blocks_per_thread) {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;
  std::vector<std::thread> threads;

  auto shift_work = [&](int bi_start, int bi_end) {
    for (int bi = bi_start; bi < bi_end; ++bi) {
      for (int bj = 0; bj < num_blocks_; ++bj) {
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
  };

  for (int t = 0; t < num_threads; ++t) {
    int start = t * blocks_per_thread;
    int end = std::min(start + blocks_per_thread, num_blocks_);
    if (start < end) {
      threads.emplace_back(shift_work, start, end);
    }
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

bool vavilov_v_cannon_stl::CannonSTL::RunImpl() {
  int num_threads = std::min(ppc::util::GetPPCNumThreads(), num_blocks_);
  int blocks_per_thread = (num_blocks_ + num_threads - 1) / num_threads;
  InitialShift(num_threads, blocks_per_thread);
  for (int iter = 0; iter < num_blocks_; ++iter) {
    BlockMultiply(num_threads, blocks_per_thread);
    ShiftBlocks(num_threads, blocks_per_thread);
  }
  return true;
}

bool vavilov_v_cannon_stl::CannonSTL::PostProcessingImpl() {
  std::ranges::copy(C_, reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
