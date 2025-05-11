#include "tbb/trubin_a_algorithm_dijkstra/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <atomic>
#include <cstddef>
#include <limits>
#include <vector>

#include "core/util/include/util.hpp"

bool trubin_a_algorithm_dijkstra_tbb::TestTaskTBB::PreProcessingImpl() {
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

bool trubin_a_algorithm_dijkstra_tbb::TestTaskTBB::ValidationImpl() {
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

void trubin_a_algorithm_dijkstra_tbb::TestTaskTBB::UpdateDistancesInBlock(
    const tbb::blocked_range<size_t>& r, std::vector<std::atomic<int>>& distances_atomic,
    std::atomic<bool>& changed_flag) {
  bool local_changed = false;
  for (size_t u = r.begin(); u < r.end(); ++u) {
    int u_dist = distances_atomic[u];
    if (u_dist == std::numeric_limits<int>::max()) {
      continue;
    }

    for (const auto& edge : adjacency_list_[u]) {
      int new_dist = u_dist + edge.weight;
      int old_dist = distances_atomic[edge.to].load();
      while (new_dist < old_dist) {
        if (distances_atomic[edge.to].compare_exchange_strong(old_dist, new_dist)) {
          local_changed = true;
          break;
        }
      }
    }
  }
  if (local_changed) {
    changed_flag = true;
  }
}

bool trubin_a_algorithm_dijkstra_tbb::TestTaskTBB::RunImpl() {
  if (num_vertices_ == 0) {
    return true;
  }

  if (start_vertex_ >= num_vertices_) {
    return false;
  }

  const int num_threads = ppc::util::GetPPCNumThreads();
  oneapi::tbb::task_arena arena(num_threads);

  arena.execute([&] {
    std::vector<std::atomic<int>> distances_atomic(num_vertices_);
    for (size_t i = 0; i < num_vertices_; ++i) {
      distances_atomic[i] = std::numeric_limits<int>::max();
    }
    distances_atomic[start_vertex_] = 0;

    std::atomic<bool> changed = true;
    while (changed) {
      changed = false;

      tbb::parallel_for(tbb::blocked_range<size_t>(0, num_vertices_, 512), [&](const tbb::blocked_range<size_t>& r) {
        UpdateDistancesInBlock(r, distances_atomic, changed);
      });
    }

    distances_.resize(num_vertices_);
    for (size_t i = 0; i < num_vertices_; ++i) {
      int d = distances_atomic[i];
      distances_[i] = (d == std::numeric_limits<int>::max()) ? -1 : d;
    }
  });

  return true;
}

bool trubin_a_algorithm_dijkstra_tbb::TestTaskTBB::PostProcessingImpl() {
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

bool trubin_a_algorithm_dijkstra_tbb::TestTaskTBB::BuildAdjacencyList(const std::vector<int>& graph_data) {
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
