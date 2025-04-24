#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shlyakov_m_shell_sort_tbb {

void ShellSort(int left, int right, std::vector<int>& arr);
void Merge(int left, int mid, int right, std::vector<int>& arr, std::vector<int>& buffer);

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
};

}  // namespace shlyakov_m_shell_sort_tbb
