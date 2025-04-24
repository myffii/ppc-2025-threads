#include "tbb/filatev_v_foks/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/mutex.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <algorithm>
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <vector>

bool filatev_v_foks_tbb::Focks::PreProcessingImpl() {
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

  auto* temp_a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* temp_b = reinterpret_cast<double*>(task_data->inputs[1]);

  for (size_t i = 0; i < size_a_.m; ++i) {
    std::copy(temp_a + (i * size_a_.n), temp_a + ((i + 1) * size_a_.n), matrix_a_.data() + (i * size_));
  }
  for (size_t i = 0; i < size_b_.m; ++i) {
    std::copy(temp_b + (i * size_b_.n), temp_b + ((i + 1) * size_b_.n), matrix_b_.data() + (i * size_));
  }

  return true;
}

bool filatev_v_foks_tbb::Focks::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[3] &&
         task_data->inputs_count[1] == task_data->inputs_count[2] &&
         task_data->outputs_count[0] == task_data->inputs_count[2] &&
         task_data->outputs_count[1] == task_data->inputs_count[1] && task_data->inputs_count[4] > 0;
}

namespace {
void ComputeBlock(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b,
                  std::vector<double>& local_block, size_t i, size_t j, size_t root, size_t size_block, size_t size) {
  for (size_t bi = 0; bi < size_block; ++bi) {
    for (size_t bj = 0; bj < size_block; ++bj) {
      for (size_t bk = 0; bk < size_block; ++bk) {
        local_block[(bi * size_block) + bj] += matrix_a[((i * size_block + bi) * size) + (root * size_block) + bk] *
                                               matrix_b[((root * size_block + bk) * size) + (j * size_block) + bj];
      }
    }
  }
}

void AccumulateResult(std::vector<double>& matrix_c, const std::vector<double>& local_block, size_t i, size_t j,
                      size_t size_block, size_t size, tbb::mutex& write_mutex) {
  tbb::mutex::scoped_lock lock(write_mutex);
  for (size_t bi = 0; bi < size_block; ++bi) {
    for (size_t bj = 0; bj < size_block; ++bj) {
      matrix_c[((i * size_block + bi) * size) + (j * size_block) + bj] += local_block[(bi * size_block) + bj];
    }
  }
}
}  // namespace

bool filatev_v_foks_tbb::Focks::RunImpl() {
  matrix_c_.assign(size_ * size_, 0);
  int grid_size = (int)(size_ / size_block_);

  oneapi::tbb::mutex write_mutex;

  const int num_threads = ppc::util::GetPPCNumThreads();
  oneapi::tbb::task_arena arena(num_threads);

  arena.execute([&] {
    oneapi::tbb::parallel_for(tbb::blocked_range<int>(0, grid_size * grid_size * grid_size),
                              [&](const tbb::blocked_range<int>& range) {
                                for (int step_i_j = range.begin(); step_i_j != range.end(); ++step_i_j) {
                                  int step = step_i_j / (grid_size * grid_size);
                                  int i = (step_i_j % (grid_size * grid_size)) / grid_size;
                                  int j = step_i_j % grid_size;
                                  size_t root = (i + step) % grid_size;

                                  std::vector<double> local_block(size_block_ * size_block_, 0);
                                  ComputeBlock(matrix_a_, matrix_b_, local_block, i, j, root, size_block_, size_);
                                  AccumulateResult(matrix_c_, local_block, i, j, size_block_, size_, write_mutex);
                                }
                              });
  });

  return true;
}

bool filatev_v_foks_tbb::Focks::PostProcessingImpl() {
  auto* temp = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < size_c_.m; ++i) {
    std::copy(matrix_c_.data() + (i * size_), matrix_c_.data() + (i * size_) + size_c_.n, temp + (i * size_c_.n));
  }
  return true;
}
