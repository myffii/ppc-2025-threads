#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace petrov_a_radix_double_batcher_omp {

class TestTaskParallelOmp : public ppc::core::Task {
 public:
  explicit TestTaskParallelOmp(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> in_;
  std::vector<double> res_;
};

}  // namespace petrov_a_radix_double_batcher_omp