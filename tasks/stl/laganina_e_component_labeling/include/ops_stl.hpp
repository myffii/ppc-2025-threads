#pragma once

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace laganina_e_component_labeling_stl {

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int m_;
  int n_;

  std::vector<int> binary_;

  void InitializeParents(std::vector<int>& parent);
  void ProcessSweep(bool reverse, std::vector<int>& parent, bool& changed) const;
  static bool UnionNodes(int a, int b, std::vector<int>& parent);
  bool CheckNeighbor(int nr, int nc, int current, std::vector<int>& parent) const;
  bool ProcessRow(int row_idx, bool reverse, std::vector<int>& parent) const;
  static int FindRoot(std::vector<int>& parent, int x);
  void FinalizeRoots(std::vector<int>& parent) const;
  void AssignLabels(std::vector<int>& parent);
  void LabelConnectedComponents();
};

inline void NormalizeLabels(std::vector<int>& vec) {
  std::vector<int> unique_labels;
  std::unordered_set<int> seen;

  for (int val : vec) {
    if (val != 0 && seen.find(val) == seen.end()) {
      unique_labels.push_back(val);
      seen.insert(val);
    }
  }

  std::unordered_map<int, int> label_map;
  int current_label = 1;
  for (int val : unique_labels) {
    label_map[val] = current_label++;
  }
  for (int& val : vec) {
    if (val != 0) {
      val = label_map[val];
    }
  }
}

}  // namespace laganina_e_component_labeling_stl