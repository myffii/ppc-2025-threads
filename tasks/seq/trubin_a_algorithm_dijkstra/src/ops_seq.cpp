#include "seq/trubin_a_algorithm_dijkstra/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

bool trubin_a_algorithm_dijkstra_seq::TestTaskSequential::PreProcessingImpl() {
  if (!validation_passed_) {
    return false;
  }

  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  std::vector<int> graph_data(in_ptr, in_ptr + input_size);

  num_vertices_ = task_data->outputs_count[0];

  if (num_vertices_ == 0 || input_size == 0) {
    return true;
  }

  adjacency_list_.assign(num_vertices_, {});
  distances_.assign(num_vertices_, std::numeric_limits<int>::max());

  if (!BuildAdjacencyList(graph_data)) {
    return false;
  }

  if (task_data->inputs.size() > 1 && task_data->inputs[1] != nullptr) {
    int* ptr = reinterpret_cast<int*>(task_data->inputs[1]);
    if (*ptr < 0 || static_cast<size_t>(*ptr) >= num_vertices_) {
      return false;
    }
    start_vertex_ = static_cast<size_t>(*ptr);
  } else {
    start_vertex_ = 0;
  }

  distances_[start_vertex_] = 0;
  return true;
}

bool trubin_a_algorithm_dijkstra_seq::TestTaskSequential::ValidationImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }

  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }

  if (task_data->outputs_count[0] == 0 && task_data->inputs_count[0] == 0) {
    validation_passed_ = true;
    return true;
  }

  if (task_data->outputs[0] == nullptr || task_data->outputs_count[0] == 0) {
    return false;
  }

  validation_passed_ = true;
  return true;
}

bool trubin_a_algorithm_dijkstra_seq::TestTaskSequential::RunImpl() {
  if (num_vertices_ == 0) {
    return true;
  }

  if (start_vertex_ >= num_vertices_) {
    return false;
  }
  if (num_vertices_ == 0 || start_vertex_ >= num_vertices_) {
    return false;
  }

  using QueueElement = std::pair<int, size_t>;
  std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<>> min_heap;
  min_heap.emplace(0, start_vertex_);

  while (!min_heap.empty()) {
    auto [current_distance, u] = min_heap.top();
    min_heap.pop();

    if (current_distance > distances_[u]) {
      continue;
    }

    for (const auto& edge : adjacency_list_[u]) {
      int new_dist = distances_[u] + edge.weight;
      if (new_dist < distances_[edge.to]) {
        distances_[edge.to] = new_dist;
        min_heap.emplace(new_dist, edge.to);
      }
    }
  }
  return true;
}

bool trubin_a_algorithm_dijkstra_seq::TestTaskSequential::PostProcessingImpl() {
  if (num_vertices_ == 0) {
    return true;
  }

  if (task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < num_vertices_; ++i) {
    out_ptr[i] = (distances_[i] == std::numeric_limits<int>::max()) ? -1 : distances_[i];
  }
  return true;
}

bool trubin_a_algorithm_dijkstra_seq::TestTaskSequential::BuildAdjacencyList(const std::vector<int>& graph_data) {
  size_t current_vertex = 0;
  size_t i = 0;

  while (i < graph_data.size()) {
    if (graph_data[i] == kEndOfVertexList) {
      if (current_vertex >= num_vertices_) {
        return false;
      }
      current_vertex++;
      i++;
      continue;
    }

    if (i + 1 >= graph_data.size()) {
      return false;
    }

    auto to = static_cast<size_t>(graph_data[i]);
    int weight = graph_data[i + 1];

    if (to >= num_vertices_ || weight < 0) {
      return false;
    }

    adjacency_list_[current_vertex].emplace_back(to, weight);
    i += 2;
  }

  return true;
}
