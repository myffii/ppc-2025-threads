#include "stl/filatev_v_foks/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool filatev_v_foks_stl::Focks::PreProcessingImpl() {
  size_block_ = task_data->inputs_count[4];
  size_a_.n = task_data->inputs_count[0];
  size_a_.m = task_data->inputs_count[1];
  size_b_.n = task_data->inputs_count[2];
  size_b_.m = task_data->inputs_count[3];

  size_c_.n = task_data->outputs_count[0];
  size_c_.m = task_data->outputs_count[1];

  size_ = std::max(size_a_.n, size_a_.m);
  size_ = std::max(size_, size_b_.n);
  size_ = std::max(size_, size_b_.m);

  size_ = (size_ % size_block_ == 0) ? size_ : ((size_ % size_block_) + 1) * size_block_;

  matrix_a_.assign(size_ * size_, 0);
  matrix_b_.assign(size_ * size_, 0);

  auto *temp_a = reinterpret_cast<double *>(task_data->inputs[0]);
  auto *temp_b = reinterpret_cast<double *>(task_data->inputs[1]);

  for (size_t i = 0; i < size_a_.m; ++i) {
    std::copy(temp_a + (i * size_a_.n), temp_a + ((i + 1) * size_a_.n), matrix_a_.data() + (i * size_));
  }
  for (size_t i = 0; i < size_b_.m; ++i) {
    std::copy(temp_b + (i * size_b_.n), temp_b + ((i + 1) * size_b_.n), matrix_b_.data() + (i * size_));
  }

  return true;
}

bool filatev_v_foks_stl::Focks::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[3] &&
         task_data->inputs_count[1] == task_data->inputs_count[2] &&
         task_data->outputs_count[0] == task_data->inputs_count[2] &&
         task_data->outputs_count[1] == task_data->inputs_count[1] && task_data->inputs_count[4] > 0;
}

void filatev_v_foks_stl::Focks::Worker(size_t start_step, size_t end_step, size_t grid_size, std::mutex &mtx) {
  for (size_t step_i_j = start_step; step_i_j < end_step; ++step_i_j) {
    size_t step = step_i_j / (grid_size * grid_size);
    size_t i = (step_i_j % (grid_size * grid_size)) / grid_size;
    size_t j = step_i_j % grid_size;

    size_t root = (i + step) % grid_size;
    std::vector<double> local_block(size_block_ * size_block_, 0);

    for (size_t bi = 0; bi < size_block_; ++bi) {
      for (size_t bj = 0; bj < size_block_; ++bj) {
        for (size_t bk = 0; bk < size_block_; ++bk) {
          local_block[(bi * size_block_) + bj] +=
              matrix_a_[((i * size_block_ + bi) * size_) + (root * size_block_) + bk] *
              matrix_b_[((root * size_block_ + bk) * size_) + (j * size_block_) + bj];
        }
      }
    }

    std::lock_guard<std::mutex> lock(mtx);
    for (size_t bi = 0; bi < size_block_; ++bi) {
      for (size_t bj = 0; bj < size_block_; ++bj) {
        matrix_c_[((i * size_block_ + bi) * size_) + (j * size_block_) + bj] += local_block[(bi * size_block_) + bj];
      }
    }
  }
}

bool filatev_v_foks_stl::Focks::RunImpl() {
  matrix_c_.assign(size_ * size_, 0);

  size_t grid_size = size_ / size_block_;
  size_t num_threads = ppc::util::GetPPCNumThreads();
  std::mutex mtx;

  if (grid_size * grid_size * grid_size >= num_threads) {
    std::vector<std::thread> threads(num_threads);

    size_t steps_per_thread = grid_size * grid_size * grid_size / num_threads;
    for (size_t t = 0; t < num_threads; ++t) {
      size_t start_step = t * steps_per_thread;
      size_t end_step = (t == num_threads - 1) ? grid_size * grid_size * grid_size : start_step + steps_per_thread;
      threads[t] = std::thread(&Focks::Worker, this, start_step, end_step, grid_size, std::ref(mtx));
    }

    for (auto &thread : threads) {
      thread.join();
    }
  } else {
    Worker(0, grid_size * grid_size * grid_size, grid_size, mtx);
  }

  return true;
}

bool filatev_v_foks_stl::Focks::PostProcessingImpl() {
  auto *temp = reinterpret_cast<double *>(task_data->outputs[0]);
  for (size_t i = 0; i < size_c_.m; ++i) {
    std::copy(matrix_c_.data() + (i * size_), matrix_c_.data() + (i * size_) + size_c_.n, temp + (i * size_c_.n));
  }
  return true;
}
