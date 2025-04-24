#pragma once

#include <cmath>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace smirnov_i_radix_sort_simple_merge_omp {

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> mas_, output_;
  static void RadixSort(std::vector<int> &mas);
  static std::vector<int> Merge(std::vector<int> mas1, std::vector<int> mas2);
};

}  // namespace smirnov_i_radix_sort_simple_merge_omp