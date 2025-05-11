#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/trubin_a_algorithm_dijkstra/include/ops_tbb.hpp"

namespace {
std::vector<int> GenerateRandomDenseGraph(size_t num_vertices, size_t max_edges_per_vertex) {
  std::vector<int> graph_data;
  std::mt19937 rng(42);
  std::uniform_int_distribution<size_t> edge_count_dist(1, max_edges_per_vertex);
  std::uniform_int_distribution<int> weight_dist(1, 10);

  for (size_t i = 0; i < num_vertices; ++i) {
    size_t num_edges = edge_count_dist(rng);
    std::unordered_set<size_t> used;
    while (used.size() < num_edges) {
      size_t to = rng() % num_vertices;
      if (to == i || used.contains(to)) {
        continue;
      }
      used.insert(to);

      graph_data.push_back(static_cast<int>(to));
      graph_data.push_back(weight_dist(rng));
    }
    graph_data.push_back(-1);
  }
  return graph_data;
}

void RunDijkstraPerfTest(size_t num_vertices, size_t max_edges_per_vertex, const std::string& run_mode) {
  auto graph_data = GenerateRandomDenseGraph(num_vertices, max_edges_per_vertex);
  int start_vertex = 0;
  std::vector<int> distances(num_vertices, -42);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.push_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_tbb->inputs_count.push_back(static_cast<uint32_t>(graph_data.size()));
  task_data_tbb->inputs.push_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_tbb->outputs.push_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_tbb->outputs_count.push_back(static_cast<uint32_t>(num_vertices));

  auto task = std::make_shared<trubin_a_algorithm_dijkstra_tbb::TestTaskTBB>(task_data_tbb);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);

  if (run_mode == "PipelineRun") {
    perf_analyzer->PipelineRun(perf_attr, perf_results);
  } else if (run_mode == "TaskRun") {
    perf_analyzer->TaskRun(perf_attr, perf_results);
  } else {
    throw std::invalid_argument("Unknown run_mode: " + run_mode);
  }

  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(distances[start_vertex], 0);
  EXPECT_TRUE(std::ranges::all_of(distances, [](int dist) { return dist >= -1; }));
}
}  // namespace

TEST(trubin_a_algorithm_dijkstra_tbb, test_pipeline_run) {
  constexpr size_t kNumVertices = 200000;
  constexpr size_t kMaxEdgesPerVertex = 100;
  RunDijkstraPerfTest(kNumVertices, kMaxEdgesPerVertex, "PipelineRun");
}

TEST(trubin_a_algorithm_dijkstra_tbb, test_task_run) {
  constexpr size_t kNumVertices = 200000;
  constexpr size_t kMaxEdgesPerVertex = 100;
  RunDijkstraPerfTest(kNumVertices, kMaxEdgesPerVertex, "TaskRun");
}
