#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/zaitsev_a_bw_labeling/include/disjoint_set.hpp"

namespace zaitsev_a_labeling_omp {

class Labeler : public ppc::core::Task {
 public:
  explicit Labeler(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<std::uint8_t> image_;
  std::vector<std::uint16_t> labels_;
  unsigned int width_;
  unsigned int height_;
  unsigned int size_;
  long chunk_;

  void LabelingRasterScan(std::vector<std::map<std::uint16_t, std::set<std::uint16_t>>>& eqs,
                          std::vector<std::uint16_t>& current_labels);
  void CalculateReplacements(std::vector<std::uint16_t>& replacements,
                             std::vector<std::map<std::uint16_t, std::set<std::uint16_t>>>& eqs,
                             std::vector<std::uint16_t>& current_label);
  void GlobalizeLabels(std::vector<std::uint16_t>& current_label);
  void UniteChunks(zaitsev_a_disjoint_set::DisjointSet<uint16_t>& dsj, std::vector<std::uint16_t>& current_label);
  void PerformReplacements(std::vector<std::uint16_t>& replacements);
};

}  // namespace zaitsev_a_labeling_omp