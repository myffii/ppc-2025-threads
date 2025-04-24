#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace gusev_n_sorting_int_simple_merging_tbb {

class SortingIntSimpleMergingTBB : public ppc::core::Task {
 public:
  explicit SortingIntSimpleMergingTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static void CountingSort(std::vector<int>& arr, int exp);
  static void RadixSortForNonNegative(std::vector<int>& arr);
  static void RadixSort(std::vector<int>& arr);
  std::vector<int> input_, output_;
};

}  // namespace gusev_n_sorting_int_simple_merging_tbb
