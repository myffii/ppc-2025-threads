#pragma once

#include <tbb/blocked_range.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace karaseva_e_congrad_tbb {

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  size_t size_{};
};

}  // namespace karaseva_e_congrad_tbb