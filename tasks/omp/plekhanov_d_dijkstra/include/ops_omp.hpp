#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace plekhanov_d_dijkstra_omp {

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static bool ConvertGraphToAdjacencyList(const std::vector<int>& graph_data, size_t num_vertices,
                                          std::vector<std::vector<std::pair<int, int>>>& graph);
  std::vector<int> graph_data_;
  std::vector<int> distances_;
  size_t start_vertex_;
  size_t num_vertices_;
  static const int kEndOfVertexList;
};

}  // namespace plekhanov_d_dijkstra_omp