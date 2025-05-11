#pragma once

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vavilov_v_cannon_stl {
class CannonSTL : public ppc::core::Task {
 public:
  explicit CannonSTL(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int N_;
  int block_size_;
  int num_blocks_;
  std::vector<double> A_;
  std::vector<double> B_;
  std::vector<double> C_;

  void InitialShift(int num_threads, int blocks_per_thread);
  void BlockMultiply(int num_threads, int blocks_per_thread);
  void ShiftBlocks(int num_threads, int blocks_per_thread);
  void ProcessSingleBlock(int bi, int bj, int bi_start, std::vector<double>& local);
  void MergeResults(int num_threads, int bi_range, const std::vector<std::vector<double>>& local_c);
};
}  // namespace vavilov_v_cannon_stl
