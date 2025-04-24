#pragma once

#include <tbb/tbb.h>

#include <cmath>
#include <deque>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "oneapi/tbb/mutex.h"

namespace smirnov_i_radix_sort_simple_merge_tbb {

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> mas_, output_;
  static void RadixSort(std::vector<int>& mas);
  static std::vector<int> Merge(std::vector<int>& mas1, std::vector<int>& mas2);
  void SortChunk(int i, int size, int nth, int& start, tbb::mutex& mtx_start, tbb::mutex& mtx_firstdq,
                 std::deque<std::vector<int>>& firstdq);
};
}  // namespace smirnov_i_radix_sort_simple_merge_tbb