#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sotskov_a_shell_sorting_with_simple_merging_tbb {

void ShellSortWithSimpleMerging(std::vector<int>& arr);
void ShellSort(std::vector<int>& arr, int left, int right);
void ParallelMerge(std::vector<int>& arr, int left, int mid, int right);

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, result_;
};
}  // namespace sotskov_a_shell_sorting_with_simple_merging_tbb