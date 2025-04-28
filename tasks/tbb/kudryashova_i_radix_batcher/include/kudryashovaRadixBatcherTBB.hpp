#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kudryashova_i_radix_batcher_tbb {
std::vector<double> GetRandomDoubleVector(int size);
void ConvertDoublesToUint64(const std::vector<double>& data, std::vector<uint64_t>& converted, size_t first);
void ConvertUint64ToDoubles(std::vector<double>& data, const std::vector<uint64_t>& converted, size_t first);
void RadixDoubleSort(std::vector<double>& data, size_t first, size_t last);
void BatcherMerge(std::vector<double>& target_array, size_t merge_start, size_t mid_point, size_t merge_end);

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_data_;
};

}  // namespace kudryashova_i_radix_batcher_tbb
