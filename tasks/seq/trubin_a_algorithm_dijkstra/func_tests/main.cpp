#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/trubin_a_algorithm_dijkstra/include/ops_seq.hpp"

namespace {
void RunDijkstraTest(const std::vector<int>& graph_data, int start_vertex, size_t vertex_count,
                     const std::vector<int>& expected) {
  std::vector<int> out(vertex_count, -42);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(graph_data.data())));
  task_data_seq->inputs_count.emplace_back(graph_data.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(vertex_count);

  trubin_a_algorithm_dijkstra_seq::TestTaskSequential task(task_data_seq);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  EXPECT_EQ(out, expected);
}

void RunDijkstraAndCheckNoExpected(const std::vector<int>& graph_data, int start_vertex, size_t num_vertices) {
  std::vector<int> out(num_vertices, -42);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(graph_data.data())));
  task_data_seq->inputs_count.emplace_back(graph_data.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(num_vertices);

  trubin_a_algorithm_dijkstra_seq::TestTaskSequential task(task_data_seq);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  ASSERT_EQ(out.size(), num_vertices);
  EXPECT_TRUE(std::ranges::all_of(out, [](int dist) { return dist >= -1; }));
}

std::vector<int> GenerateRandomGraph(size_t num_vertices, size_t max_edges_per_vertex, int max_weight) {
  std::vector<int> graph_data;
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> vertex_dist(0, static_cast<int>(num_vertices - 1));
  std::uniform_int_distribution<int> weight_dist(1, max_weight);
  std::uniform_int_distribution<int> edge_count_dist(0, static_cast<int>(max_edges_per_vertex));

  for (size_t v = 0; v < num_vertices; ++v) {
    int num_edges = edge_count_dist(rng);
    for (int e = 0; e < num_edges; ++e) {
      int to = vertex_dist(rng);
      int weight = weight_dist(rng);
      graph_data.push_back(to);
      graph_data.push_back(weight);
    }
    graph_data.push_back(-1);
  }
  return graph_data;
}

std::vector<int> GenerateRandomDenseGraph(size_t num_vertices, int max_weight) {
  std::vector<int> graph_data;
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> weight_dist(1, max_weight);

  for (size_t v = 0; v < num_vertices; ++v) {
    for (size_t to = 0; to < num_vertices; ++to) {
      if (v != to && (to + v) % 2 == 0) {
        int weight = weight_dist(rng);
        graph_data.push_back(static_cast<int>(to));
        graph_data.push_back(weight);
      }
    }
    graph_data.push_back(-1);
  }
  return graph_data;
}

}  // namespace

TEST(trubin_a_algorithm_dijkstra_seq, trivial_graph) { RunDijkstraTest({-1}, 0, 1, {0}); }

TEST(trubin_a_algorithm_dijkstra_seq, linear_graph) {
  RunDijkstraTest({1, 2, -1, 2, 3, -1, 3, 1, -1, -1}, 0, 4, {0, 2, 5, 6});
}

TEST(trubin_a_algorithm_dijkstra_seq, disconnected_graph) { RunDijkstraTest({1, 1, -1, -1, -1}, 0, 3, {0, 1, -1}); }

TEST(trubin_a_algorithm_dijkstra_seq, cycle_graph) { RunDijkstraTest({1, 2, -1, 2, 2, -1, 0, 2, -1}, 0, 3, {0, 2, 4}); }

TEST(trubin_a_algorithm_dijkstra_seq, fork_graph) { RunDijkstraTest({1, 1, 2, 5, -1, 2, 1, -1, -1}, 0, 3, {0, 1, 2}); }

TEST(trubin_a_algorithm_dijkstra_seq, multiple_paths_same_cost) {
  RunDijkstraTest({1, 2, 2, 2, -1, 3, 2, -1, 3, 2, -1, -1}, 0, 4, {0, 2, 2, 4});
}

TEST(trubin_a_algorithm_dijkstra_seq, unreachable_from_middle) {
  RunDijkstraTest({1, 1, -1, 2, 1, -1, -1, -1}, 0, 4, {0, 1, 2, -1});
}

TEST(trubin_a_algorithm_dijkstra_seq, empty_graph) {
  std::vector<int> empty_graph;
  std::vector<int> expected;
  RunDijkstraTest(empty_graph, 0, 0, expected);
}

TEST(trubin_a_algorithm_dijkstra_seq, invalid_start_vertex) {
  std::vector<int> graph = {1, 1, -1, -1};
  std::vector<int> out(2, -1);
  int start_vertex = 10;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  task_data_seq->inputs_count.emplace_back(graph.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(2);

  trubin_a_algorithm_dijkstra_seq::TestTaskSequential task(task_data_seq);
  EXPECT_TRUE(task.Validation());
  EXPECT_FALSE(task.PreProcessing());
}

TEST(trubin_a_algorithm_dijkstra_seq, negative_weight_should_fail) {
  std::vector<int> graph = {1, -1, -1, -1};
  std::vector<int> out(2, -1);
  int start_vertex = 0;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  task_data_seq->inputs_count.emplace_back(graph.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(2);

  trubin_a_algorithm_dijkstra_seq::TestTaskSequential task(task_data_seq);
  EXPECT_TRUE(task.Validation());
  EXPECT_FALSE(task.PreProcessing());
}

TEST(trubin_a_algorithm_dijkstra_seq, validation_passes_on_empty_graph_and_outputs) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(nullptr);
  task_data_seq->inputs_count.emplace_back(0);
  task_data_seq->outputs.emplace_back(nullptr);
  task_data_seq->outputs_count.emplace_back(0);
  trubin_a_algorithm_dijkstra_seq::TestTaskSequential task(task_data_seq);
  EXPECT_TRUE(task.Validation());
}

TEST(trubin_a_algorithm_dijkstra_seq, validation_fails_on_empty_counts) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(nullptr);
  task_data_seq->outputs.emplace_back(nullptr);
  trubin_a_algorithm_dijkstra_seq::TestTaskSequential task(task_data_seq);
  EXPECT_FALSE(task.Validation());
}

TEST(trubin_a_algorithm_dijkstra_seq, validation_fails_on_empty_inputs) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  trubin_a_algorithm_dijkstra_seq::TestTaskSequential task(task_data_seq);
  EXPECT_FALSE(task.Validation());
}

TEST(trubin_a_algorithm_dijkstra_seq, multi_component_graph) {
  RunDijkstraTest({1, 1, -1, -1, 3, 2, -1, -1}, 0, 4, {0, 1, -1, -1});
}

TEST(trubin_a_algorithm_dijkstra_seq, start_from_nonzero_vertex) {
  RunDijkstraTest({1, 3, -1, 2, 4, -1, 3, 2, -1, -1}, 2, 4, {-1, -1, 0, 2});
}

TEST(trubin_a_algorithm_dijkstra_seq, random_graph) {
  const size_t num_vertices = 20;
  const size_t max_edges_per_vertex = 5;
  const int max_weight = 20;

  auto graph_data = GenerateRandomGraph(num_vertices, max_edges_per_vertex, max_weight);
  RunDijkstraAndCheckNoExpected(graph_data, 0, num_vertices);
}

TEST(trubin_a_algorithm_dijkstra_seq, random_dense_graph) {
  const size_t num_vertices = 30;
  const int max_weight = 15;

  auto graph_data = GenerateRandomDenseGraph(num_vertices, max_weight);
  RunDijkstraAndCheckNoExpected(graph_data, static_cast<int>(num_vertices / 2), num_vertices);
}
