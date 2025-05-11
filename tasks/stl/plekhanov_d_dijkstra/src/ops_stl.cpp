#include "stl/plekhanov_d_dijkstra/include/ops_stl.hpp"

#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

namespace plekhanov_d_dijkstra_stl {

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

}  // namespace plekhanov_d_dijkstra_stl

const int plekhanov_d_dijkstra_stl::TestTaskSTL::kEndOfVertexList = -1;

bool plekhanov_d_dijkstra_stl::TestTaskSTL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  graph_data_.assign(in_ptr, in_ptr + input_size);
  num_vertices_ = task_data->outputs_count[0];
  distances_.assign(num_vertices_, INT_MAX);

  if (task_data->inputs.size() > 1 && task_data->inputs[1] != nullptr) {
    start_vertex_ = *reinterpret_cast<int*>(task_data->inputs[1]);
  } else {
    start_vertex_ = 0;
  }
  distances_[start_vertex_] = 0;
  return true;
}

bool plekhanov_d_dijkstra_stl::TestTaskSTL::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0 && !task_data->outputs_count.empty() &&
         task_data->outputs_count[0] > 0;
}

bool plekhanov_d_dijkstra_stl::TestTaskSTL::RunImpl() {
  std::vector<std::vector<std::pair<int, int>>> graph(num_vertices_);

  if (!ConvertGraphToAdjacencyList(graph_data_, num_vertices_, graph)) {
    return false;
  }

  struct Compare {
    bool operator()(const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.first > b.first; }
  };

  std::vector<std::atomic<int>> distance(num_vertices_);
  for (auto& d : distance) {
    d.store(INT_MAX);
  }
  distance[start_vertex_] = 0;

  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, Compare> pq;
  pq.emplace(0, start_vertex_);

  std::mutex pq_mutex;

  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  auto worker = [&]() {
    while (true) {
      int u = -1;
      int cur_dist = -1;

      {
        std::lock_guard<std::mutex> lock(pq_mutex);
        if (!pq.empty()) {
          auto top = pq.top();
          pq.pop();
          u = top.second;
          cur_dist = top.first;
        } else {
          break;
        }
      }

      for (const auto& [v, weight] : graph[u]) {
        int new_dist = cur_dist + weight;
        int old_val = distance[v].load();

        while (new_dist < old_val && !distance[v].compare_exchange_weak(old_val, new_dist)) {
        }

        if (new_dist < old_val) {
          std::lock_guard<std::mutex> lock(pq_mutex);
          pq.emplace(new_dist, v);
        }
      }
    }
  };

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  distances_.resize(num_vertices_);
  for (size_t k = 0; k < num_vertices_; ++k) {
    distances_[k] = distance[k].load();
  }

  return true;
}

bool plekhanov_d_dijkstra_stl::TestTaskSTL::PostProcessingImpl() {
  auto* output = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < distances_.size(); ++i) {
    output[i] = distances_[i];
  }
  return true;
}