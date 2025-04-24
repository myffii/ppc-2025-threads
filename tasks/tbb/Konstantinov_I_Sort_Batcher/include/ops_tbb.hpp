#pragma once

#include <cmath>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konstantinov_i_sort_batcher_tbb {

class RadixSortBatcherTBB : public ppc::core::Task {
 public:
  explicit RadixSortBatcherTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> mas_, output_;
};
}  // namespace konstantinov_i_sort_batcher_tbb