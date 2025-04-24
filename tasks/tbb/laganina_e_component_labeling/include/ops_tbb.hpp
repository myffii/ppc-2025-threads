#pragma once

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/parallel_for.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace laganina_e_component_labeling_tbb {

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;

  bool PreProcessingImpl() override;

  bool PostProcessingImpl() override;
  bool RunImpl() override;

 private:
  int rows_;
  int cols_;
  std::vector<int> data_;

  struct UnionFind {
    std::vector<int> parent;

    UnionFind(int size, std::vector<int>& data) : parent(size) {
      tbb::parallel_for(0, size, [&](int i) { parent[i] = data[i] ? i : -1; });
    }

    int Find(int x);
    void Unite(int x, int y);
  };
  void ProcessComponents(UnionFind& uf);
  void AssignFinalLabels(int size, UnionFind uf);
  void ProcessRange(const tbb::blocked_range2d<int>& range, UnionFind& uf);
  void ProcessRow(int row, const tbb::blocked_range<int>& col_range, UnionFind& uf);
  void CheckAllNeighbors(int row, int col, int idx, UnionFind& uf);
  void LabelComponents();
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

}  // namespace laganina_e_component_labeling_tbb
