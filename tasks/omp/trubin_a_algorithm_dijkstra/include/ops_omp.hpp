#pragma once

#include <atomic>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace trubin_a_algorithm_dijkstra_omp {
struct Edge {
  size_t to;
  int weight;

  Edge(size_t to, int weight) : to(to), weight(weight) {}
};

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  bool BuildAdjacencyList(const std::vector<int>& graph_data);
  void ProcessCurrentBucket(std::vector<int>& current, std::vector<std::vector<int>>& buckets, int delta,
                            std::vector<std::atomic<int>>& distances_atomic);
  void ProcessSingleVertex(int u, int delta, std::vector<int>& next_bucket,
                           std::unordered_map<size_t, std::vector<int>>& heavy_buckets,
                           std::vector<std::atomic<int>>& distances_atomic);
  std::vector<std::vector<Edge>> adjacency_list_;
  std::vector<int> distances_;
  size_t start_vertex_ = 0;
  size_t num_vertices_ = 0;

  bool validation_passed_ = false;

  static constexpr int kEndOfVertexList = -1;
};

}  // namespace trubin_a_algorithm_dijkstra_omp