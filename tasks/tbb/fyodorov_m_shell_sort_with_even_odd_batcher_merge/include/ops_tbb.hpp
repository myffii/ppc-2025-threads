#pragma once

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/tbb.h>

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb {

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void CallBatcherMerge(std::vector<int>& left, std::vector<int>& right, std::vector<int>& result) {
    BatcherMerge(left, right, result);
  }

 private:
  std::vector<int> input_, output_;

  void ShellSort();
  static void BatcherMerge(std::vector<int>& left, std::vector<int>& right, std::vector<int>& result);
};

}  // namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb
