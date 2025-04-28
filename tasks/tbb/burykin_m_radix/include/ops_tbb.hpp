#pragma once

#include <array>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace burykin_m_radix_tbb {

class RadixTBB : public ppc::core::Task {
 public:
  explicit RadixTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::array<int, 256> ComputeFrequencyParallel(const std::vector<int>& a, int shift);
  static std::array<int, 256> ComputeIndices(const std::array<int, 256>& count);
  static void DistributeElementsParallel(const std::vector<int>& a, std::vector<int>& b,
                                         const std::array<int, 256>& index, int shift);

 private:
  std::vector<int> input_, output_;
};

}  // namespace burykin_m_radix_tbb