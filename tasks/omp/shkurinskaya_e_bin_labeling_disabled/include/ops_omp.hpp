#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shkurinskaya_e_bin_labeling_omp {

class TaskOMP : public ppc::core::Task {
 public:
  explicit TaskOMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int width_, height_;
  std::vector<int> input_;
  std::vector<int> res_;
  std::vector<int> parent_;
  std::vector<int> label_;
  std::vector<int> rank_;
  void UnionSets(int index_a, int index_b);
  int FindRoot(int index);
  [[nodiscard]] bool IsValidIndex(int i, int j) const;
  void ProcessUnion();
};

}  // namespace shkurinskaya_e_bin_labeling_omp
