#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb {
class ShellSortTBB : public ppc::core::Task {
 public:
  explicit ShellSortTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int c_threads_;
  int mini_batch_;
  int size_;
  int n_;
  std::vector<int> array_;
  std::vector<int> mass_;

  void ParallelShellSort();
  void ShellSort(int start);
  void Merge();
  void LastMerge();
  void MergeBlocks(int id_l, int id_r, int len);
  void FindThreadVariables();
};

}  // namespace volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb