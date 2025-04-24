#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;

  void ShellSort();
  static void BatcherMerge(std::vector<int>& left, std::vector<int>& right, std::vector<int>& result);
};

}  // namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_seq