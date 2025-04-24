#pragma once

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kudryashova_i_radix_batcher_omp {
std::vector<double> GetRandomDoubleVector(int size);
void RadixDoubleSort(std::vector<double>& data, size_t first, size_t last);
void BatcherMerge(std::vector<double>& target_array, size_t merge_start, size_t mid_point, size_t merge_end);

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_data_;
};

}  // namespace kudryashova_i_radix_batcher_omp
