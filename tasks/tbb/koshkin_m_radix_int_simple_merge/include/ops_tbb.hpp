#pragma once

#include <span>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace koshkin_m_radix_int_simple_merge {

class TbbT : public ppc::core::Task {
 public:
  explicit TbbT(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::span<int> in_;
  std::vector<std::vector<int>> blocks_;
  std::vector<int> out_;
};

}  // namespace koshkin_m_radix_int_simple_merge