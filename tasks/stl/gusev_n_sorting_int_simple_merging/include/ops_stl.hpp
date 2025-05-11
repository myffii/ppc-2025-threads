#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace gusev_n_sorting_int_simple_merging_stl {

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  static constexpr int kDigitCount = 10;

  static void RadixSort(std::vector<int>& arr);
  static void SplitAndSort(std::vector<int>& arr, std::vector<int>& negatives, std::vector<int>& positives);
  static void RadixSortForNonNegative(std::vector<int>& arr);
  static void CountingSort(std::vector<int>& arr, int exp);
};

}  // namespace gusev_n_sorting_int_simple_merging_stl
