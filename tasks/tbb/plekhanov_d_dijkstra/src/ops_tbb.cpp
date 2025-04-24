#include "tbb/plekhanov_d_dijkstra/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/mutex.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

#include "tbb/tbb.h"

namespace plekhanov_d_dijkstra_tbb {

int plekhanov_d_dijkstra_tbb::TestTaskTBB::FindMinDistanceVertexTBB(const std::vector<bool>& visited) const {
  return oneapi::tbb::parallel_reduce(
      oneapi::tbb::blocked_range<int>(0, static_cast<int>(num_vertices_)), -1,
      [&](const oneapi::tbb::blocked_range<int>& r, int local_u) -> int {
        int local_min = std::numeric_limits<int>::max();
        int current_u = -1;
        for (int v = r.begin(); v < r.end(); ++v) {
          if (!visited[v] && distances_[v] < local_min) {
            local_min = distances_[v];
            current_u = v;
          }
        }
        if (local_u == -1) {
          return current_u;
        }
        if (current_u == -1) {
          return local_u;
        }
        return (distances_[current_u] < distances_[local_u]) ? current_u : local_u;
      },
      [&](int u1, int u2) -> int {
        if (u1 == -1) {
          return u2;
        }
        if (u2 == -1) {
          return u1;
        }
        return (distances_[u1] < distances_[u2]) ? u1 : u2;
      });
}

void plekhanov_d_dijkstra_tbb::TestTaskTBB::RelaxEdgesTBB(int u,
                                                          const std::vector<std::vector<std::pair<int, int>>>& graph,
                                                          const std::vector<bool>& visited) {
  oneapi::tbb::parallel_for(0, static_cast<int>(graph[u].size()), [&](int j) {
    int v = graph[u][j].first;
    int weight = graph[u][j].second;
    if (!visited[v] && distances_[u] != std::numeric_limits<int>::max()) {
      int new_dist = distances_[u] + weight;
      oneapi::tbb::mutex::scoped_lock lock(mutex_);
      distances_[v] = std::min(new_dist, distances_[v]);
    }
  });
}

namespace {

bool ConvertGraphToAdjacencyList(const std::vector<int>& graph_data, size_t num_vertices,
                                 std::vector<std::vector<std::pair<int, int>>>& graph) {
  graph.assign(num_vertices, {});
  size_t current_vertex = 0;
  size_t i = 0;
  while (i < graph_data.size() && current_vertex < num_vertices) {
    if (graph_data[i] == -1) {
      current_vertex++;
      i++;
      continue;
    }
    if (i + 1 >= graph_data.size()) {
      break;
    }
    size_t dest = graph_data[i];
    int weight = graph_data[i + 1];
    if (weight < 0) {
      return false;
    }
    if (dest < num_vertices) {
      graph[current_vertex].emplace_back(static_cast<int>(dest), weight);
    }
    i += 2;
  }
  return true;
}

}  // namespace

}  // namespace plekhanov_d_dijkstra_tbb

const int plekhanov_d_dijkstra_tbb::TestTaskTBB::kEndOfVertexList = -1;

bool plekhanov_d_dijkstra_tbb::TestTaskTBB::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  graph_data_.assign(in_ptr, in_ptr + input_size);
  num_vertices_ = task_data->outputs_count[0];
  distances_.assign(num_vertices_, INT_MAX);
  distances_.resize(num_vertices_);

  if (task_data->inputs.size() > 1 && task_data->inputs[1] != nullptr) {
    start_vertex_ = *reinterpret_cast<int*>(task_data->inputs[1]);
  } else {
    start_vertex_ = 0;
  }
  distances_[start_vertex_] = 0;
  return true;
}

bool plekhanov_d_dijkstra_tbb::TestTaskTBB::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0 && !task_data->outputs_count.empty() &&
         task_data->outputs_count[0] > 0;
}

bool plekhanov_d_dijkstra_tbb::TestTaskTBB::RunImpl() {
  std::vector<std::vector<std::pair<int, int>>> graph(num_vertices_);
  if (!ConvertGraphToAdjacencyList(graph_data_, num_vertices_, graph)) {
    return false;
  }

  std::vector<bool> visited(num_vertices_, false);
  for (int count = 0; count < static_cast<int>(num_vertices_) - 1; ++count) {
    int u = FindMinDistanceVertexTBB(visited);
    if (u == -1) {
      break;
    }

    visited[u] = true;
    RelaxEdgesTBB(u, graph, visited);
  }

  return true;
}

bool plekhanov_d_dijkstra_tbb::TestTaskTBB::PostProcessingImpl() {
  auto* output = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < distances_.size(); ++i) {
    output[i] = distances_[i];
  }
  return true;
}
