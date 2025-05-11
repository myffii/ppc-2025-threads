#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace trubin_a_algorithm_dijkstra_seq {
struct Edge {
  size_t to;
  int weight;

  Edge(size_t to, int weight) : to(to), weight(weight) {}
};

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  bool BuildAdjacencyList(const std::vector<int>& graph_data);

  std::vector<std::vector<Edge>> adjacency_list_;
  std::vector<int> distances_;
  size_t start_vertex_ = 0;
  size_t num_vertices_ = 0;

  bool validation_passed_ = false;

  static constexpr int kEndOfVertexList = -1;
};

}  // namespace trubin_a_algorithm_dijkstra_seq
