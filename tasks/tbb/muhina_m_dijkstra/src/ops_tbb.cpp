#include "tbb/muhina_m_dijkstra/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/concurrent_priority_queue.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/spin_mutex.h>

#include <climits>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

const int muhina_m_dijkstra_tbb::TestTaskTBB::kEndOfVertexList = -1;

namespace {
void RunDijkstraAlgorithm(const std::vector<std::vector<std::pair<size_t, int>>> &adj_list, std::vector<int> &distances,
                          size_t start_vertex) {
  oneapi::tbb::concurrent_priority_queue<std::pair<int, size_t>, std::greater<>> pq;
  pq.push({0, start_vertex});
  oneapi::tbb::spin_mutex mutex;

  while (true) {
    std::pair<int, size_t> top;
    if (!pq.try_pop(top)) {
      if (pq.empty()) {
        break;
      }
      continue;
    }

    size_t u = top.second;
    int dist_u = top.first;

    if (dist_u > distances[u]) {
      continue;
    }

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, adj_list[u].size()),
                              [&](const oneapi::tbb::blocked_range<size_t> &r) {
                                for (size_t i = r.begin(); i != r.end(); ++i) {
                                  size_t v = adj_list[u][i].first;
                                  int weight = adj_list[u][i].second;
                                  int new_dist = distances[u] + weight;

                                  if (new_dist < distances[v]) {
                                    oneapi::tbb::spin_mutex::scoped_lock lock(mutex);
                                    if (new_dist < distances[v]) {
                                      distances[v] = new_dist;
                                      pq.push({new_dist, v});
                                    }
                                  }
                                }
                              });
  }
}
}  // namespace

bool muhina_m_dijkstra_tbb::TestTaskTBB::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  graph_data_.assign(in_ptr, in_ptr + input_size);

  num_vertices_ = task_data->outputs_count[0];
  distances_.resize(num_vertices_);
  for (size_t i = 0; i < num_vertices_; ++i) {
    distances_[i] = INT_MAX;
  }
  if (task_data->inputs.size() > 1 && task_data->inputs[1] != nullptr) {
    start_vertex_ = *reinterpret_cast<int *>(task_data->inputs[1]);
  } else {
    start_vertex_ = 0;
  }
  distances_[start_vertex_] = 0;

  return true;
}

bool muhina_m_dijkstra_tbb::TestTaskTBB::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0 && !task_data->outputs_count.empty() &&
         task_data->outputs_count[0] > 0;
}

bool muhina_m_dijkstra_tbb::TestTaskTBB::RunImpl() {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list(num_vertices_);
  size_t current_vertex = 0;
  size_t i = 0;

  while (i < graph_data_.size() && current_vertex < num_vertices_) {
    if (graph_data_[i] == kEndOfVertexList) {
      current_vertex++;
      i++;
      continue;
    }

    if (i + 1 >= graph_data_.size()) {
      break;
    }

    size_t dest = graph_data_[i];
    int weight = graph_data_[i + 1];
    if (weight < 0) {
      return false;
    }

    if (dest < num_vertices_) {
      adj_list[current_vertex].emplace_back(dest, weight);
    }

    i += 2;
  }
  RunDijkstraAlgorithm(adj_list, distances_, start_vertex_);

  return true;
}

bool muhina_m_dijkstra_tbb::TestTaskTBB::PostProcessingImpl() {
  for (size_t i = 0; i < distances_.size(); ++i) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = distances_[i];
  }
  return true;
}
