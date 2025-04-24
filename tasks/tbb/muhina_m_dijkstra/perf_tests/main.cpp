#include <gtest/gtest.h>

#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/muhina_m_dijkstra/include/ops_tbb.hpp"

namespace {

std::vector<std::vector<std::pair<size_t, int>>> GenerateLargeGraph(size_t k_num_vertices) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list(k_num_vertices);
  for (size_t i = 0; i < k_num_vertices; ++i) {
    for (size_t j = 0; j < k_num_vertices; ++j) {
      if (i != j) {
        if (rand() % 3 == 0) {
          int weight = (rand() % 10) + 1;
          adj_list[i].emplace_back(j, weight);
        }
      }
    }
  }
  return adj_list;
}

std::vector<int> ConvertGraphToData(const std::vector<std::vector<std::pair<size_t, int>>>& adj_list) {
  std::vector<int> graph_data;
  for (const auto& vertex_edges : adj_list) {
    for (const auto& edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }
  return graph_data;
}

std::vector<int> Dijkstra(const std::vector<std::vector<std::pair<size_t, int>>>& adj_list, size_t start_vertex) {
  const size_t num_vertices = adj_list.size();
  std::vector<int> distances(num_vertices, INT_MAX);
  distances[start_vertex] = 0;

  std::priority_queue<std::pair<int, size_t>, std::vector<std::pair<int, size_t>>, std::greater<>> pq;
  pq.emplace(0, start_vertex);

  while (!pq.empty()) {
    size_t u = pq.top().second;
    int dist_u = pq.top().first;
    pq.pop();

    if (dist_u > distances[u]) {
      continue;
    }

    for (const auto& edge : adj_list[u]) {
      size_t v = edge.first;
      int weight = edge.second;

      if (distances[u] != INT_MAX && distances[u] + weight < distances[v]) {
        distances[v] = distances[u] + weight;
        pq.emplace(distances[v], v);
      }
    }
  }

  return distances;
}
}  // namespace

TEST(muhina_m_dijkstra_tbb, test_pipeline_run) {
  constexpr size_t kNumVertices = 5000;
  size_t start_vertex = 0;

  auto adj_list = GenerateLargeGraph(kNumVertices);
  auto graph_data = ConvertGraphToData(adj_list);
  auto expected_distances = Dijkstra(adj_list, start_vertex);
  std::vector<int> distances(kNumVertices, INT_MAX);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_tbb->inputs_count.emplace_back(graph_data.size());

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_tbb->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_tbb->outputs_count.emplace_back(kNumVertices);

  auto test_task_tbb = std::make_shared<muhina_m_dijkstra_tbb::TestTaskTBB>(task_data_tbb);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (size_t i = 0; i < kNumVertices; ++i) {
    EXPECT_EQ(distances[i], expected_distances[i]);
  }
}

TEST(muhina_m_dijkstra_tbb, test_task_run) {
  constexpr size_t kNumVertices = 5000;
  size_t start_vertex = 0;

  auto adj_list = GenerateLargeGraph(kNumVertices);
  auto graph_data = ConvertGraphToData(adj_list);
  auto expected_distances = Dijkstra(adj_list, start_vertex);
  std::vector<int> distances(kNumVertices, INT_MAX);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_tbb->inputs_count.emplace_back(graph_data.size());

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_tbb->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_tbb->outputs_count.emplace_back(kNumVertices);

  auto test_task_tbb = std::make_shared<muhina_m_dijkstra_tbb::TestTaskTBB>(task_data_tbb);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (size_t i = 0; i < kNumVertices; ++i) {
    EXPECT_EQ(distances[i], expected_distances[i]);
  }
}
