#pragma once

#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace filatev_v_foks_stl {

struct MatrixSize {
  size_t n;
  size_t m;

  MatrixSize() {
    n = 0;
    m = 0;
  }
  MatrixSize(size_t n, size_t m) : n(n), m(m) {}
};

class Focks : public ppc::core::Task {
 public:
  explicit Focks(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void Worker(size_t start_step, size_t end_step, size_t grid_size, std::mutex& mtx);

 private:
  MatrixSize size_a_;
  MatrixSize size_b_;
  MatrixSize size_c_;

  size_t size_block_{};
  size_t size_{};

  std::vector<double> matrix_a_;
  std::vector<double> matrix_b_;
  std::vector<double> matrix_c_;
};

}  // namespace filatev_v_foks_stl