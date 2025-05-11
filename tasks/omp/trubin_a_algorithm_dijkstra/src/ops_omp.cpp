#include "omp/trubin_a_algorithm_dijkstra/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

bool trubin_a_algorithm_dijkstra_omp::TestTaskOpenMP::PreProcessingImpl() {
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

bool trubin_a_algorithm_dijkstra_omp::TestTaskOpenMP::ValidationImpl() {
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

bool trubin_a_algorithm_dijkstra_omp::TestTaskOpenMP::RunImpl() {
  if (num_vertices_ == 0) {
    return true;
  }

  if (start_vertex_ >= num_vertices_) {
    return false;
  }

  const int delta = 5;

  std::vector<std::atomic<int>> distances_atomic(num_vertices_);
  for (size_t i = 0; i < num_vertices_; ++i) {
    distances_atomic[i].store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
  }
  distances_atomic[start_vertex_].store(0, std::memory_order_relaxed);

  std::vector<std::vector<int>> buckets(1);
  buckets[0].push_back(static_cast<int>(start_vertex_));

  while (true) {
    auto it = std::ranges::find_if(buckets, [](const auto& bucket) { return !bucket.empty(); });
    if (it == buckets.end()) {
      break;
    }

    std::vector<int> current;
    std::swap(current, *it);

    ProcessCurrentBucket(current, buckets, delta, distances_atomic);
  }

  for (size_t i = 0; i < num_vertices_; ++i) {
    distances_[i] = distances_atomic[i].load(std::memory_order_relaxed);
  }

  return true;
}

bool trubin_a_algorithm_dijkstra_omp::TestTaskOpenMP::PostProcessingImpl() {
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

bool trubin_a_algorithm_dijkstra_omp::TestTaskOpenMP::BuildAdjacencyList(const std::vector<int>& graph_data) {
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
void trubin_a_algorithm_dijkstra_omp::TestTaskOpenMP::ProcessCurrentBucket(
    std::vector<int>& current, std::vector<std::vector<int>>& buckets, const int delta,
    std::vector<std::atomic<int>>& distances_atomic) {
  auto num_threads = static_cast<size_t>(ppc::util::GetPPCNumThreads());
  std::vector<std::vector<int>> local_next(num_threads);
  std::vector<std::unordered_map<size_t, std::vector<int>>> local_heavy(num_threads);

  while (!current.empty()) {
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      auto& next_bucket = local_next[tid];
      auto& heavy_buckets = local_heavy[tid];

#pragma omp for schedule(dynamic)
      for (int i = 0; i < static_cast<int>(current.size()); ++i) {
        ProcessSingleVertex(current[i], delta, next_bucket, heavy_buckets, distances_atomic);
      }
    }

    current.clear();
    for (auto& next : local_next) {
      current.insert(current.end(), std::make_move_iterator(next.begin()), std::make_move_iterator(next.end()));
      next.clear();
    }

    for (auto& heavy_map : local_heavy) {
      for (auto& [bucket_idx, verts] : heavy_map) {
        if (bucket_idx >= buckets.size()) {
          buckets.resize(bucket_idx + 1);
        }
        buckets[bucket_idx].insert(buckets[bucket_idx].end(), std::make_move_iterator(verts.begin()),
                                   std::make_move_iterator(verts.end()));
        verts.clear();
      }
    }
  }
}

void trubin_a_algorithm_dijkstra_omp::TestTaskOpenMP::ProcessSingleVertex(
    int u, int delta, std::vector<int>& next_bucket, std::unordered_map<size_t, std::vector<int>>& heavy_buckets,
    std::vector<std::atomic<int>>& distances_atomic) {
  int dist_u = distances_atomic[u].load(std::memory_order_relaxed);

  for (const auto& edge : adjacency_list_[u]) {
    int v = static_cast<int>(edge.to);
    int weight = edge.weight;
    int new_dist = dist_u + weight;

    int old_dist = distances_atomic[v].load(std::memory_order_relaxed);
    while (new_dist < old_dist) {
      if (distances_atomic[v].compare_exchange_weak(old_dist, new_dist, std::memory_order_relaxed)) {
        if (weight <= delta) {
          next_bucket.push_back(v);
        } else {
          size_t bucket_idx = new_dist / delta;
          heavy_buckets[bucket_idx].push_back(v);
        }
        break;
      }
    }
  }
}
