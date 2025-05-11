#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/trubin_a_algorithm_dijkstra/include/ops_omp.hpp"

namespace {
void RunDijkstraTest(const std::vector<int>& graph_data, int start_vertex, size_t vertex_count,
                     const std::vector<int>& expected) {
  std::vector<int> out(vertex_count, -42);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(graph_data.data())));
  task_data_omp->inputs_count.emplace_back(graph_data.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(vertex_count);

  trubin_a_algorithm_dijkstra_omp::TestTaskOpenMP task(task_data_omp);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(out, expected);
}
}  // namespace

TEST(trubin_a_algorithm_dijkstra_omp, trivial_graph) { RunDijkstraTest({-1}, 0, 1, {0}); }

TEST(trubin_a_algorithm_dijkstra_omp, linear_graph) {
  RunDijkstraTest({1, 2, -1, 2, 3, -1, 3, 1, -1, -1}, 0, 4, {0, 2, 5, 6});
}

TEST(trubin_a_algorithm_dijkstra_omp, disconnected_graph) { RunDijkstraTest({1, 1, -1, -1, -1}, 0, 3, {0, 1, -1}); }

TEST(trubin_a_algorithm_dijkstra_omp, cycle_graph) { RunDijkstraTest({1, 2, -1, 2, 2, -1, 0, 2, -1}, 0, 3, {0, 2, 4}); }

TEST(trubin_a_algorithm_dijkstra_omp, fork_graph) { RunDijkstraTest({1, 1, 2, 5, -1, 2, 1, -1, -1}, 0, 3, {0, 1, 2}); }

TEST(trubin_a_algorithm_dijkstra_omp, multiple_paths_same_cost) {
  RunDijkstraTest({1, 2, 2, 2, -1, 3, 2, -1, 3, 2, -1, -1}, 0, 4, {0, 2, 2, 4});
}

TEST(trubin_a_algorithm_dijkstra_omp, unreachable_from_middle) {
  RunDijkstraTest({1, 1, -1, 2, 1, -1, -1, -1}, 0, 4, {0, 1, 2, -1});
}

TEST(trubin_a_algorithm_dijkstra_omp, empty_graph) {
  std::vector<int> empty_graph;
  std::vector<int> expected;
  RunDijkstraTest(empty_graph, 0, 0, expected);
}

TEST(trubin_a_algorithm_dijkstra_omp, invalid_start_vertex) {
  std::vector<int> graph = {1, 1, -1, -1};
  std::vector<int> out(2, -1);
  int start_vertex = 10;

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  task_data_omp->inputs_count.emplace_back(graph.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(2);

  trubin_a_algorithm_dijkstra_omp::TestTaskOpenMP task(task_data_omp);
  EXPECT_TRUE(task.Validation());
  EXPECT_FALSE(task.PreProcessing());
}

TEST(trubin_a_algorithm_dijkstra_omp, negative_weight_should_fail) {
  std::vector<int> graph = {1, -1, -1, -1};
  std::vector<int> out(2, -1);
  int start_vertex = 0;

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  task_data_omp->inputs_count.emplace_back(graph.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(2);

  trubin_a_algorithm_dijkstra_omp::TestTaskOpenMP task(task_data_omp);
  EXPECT_TRUE(task.Validation());
  EXPECT_FALSE(task.PreProcessing());
}

TEST(trubin_a_algorithm_dijkstra_omp, validation_passes_on_empty_graph_and_outputs) {
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(nullptr);
  task_data_omp->inputs_count.emplace_back(0);
  task_data_omp->outputs.emplace_back(nullptr);
  task_data_omp->outputs_count.emplace_back(0);
  trubin_a_algorithm_dijkstra_omp::TestTaskOpenMP task(task_data_omp);
  EXPECT_TRUE(task.Validation());
}

TEST(trubin_a_algorithm_dijkstra_omp, validation_fails_on_empty_counts) {
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(nullptr);
  task_data_omp->outputs.emplace_back(nullptr);
  trubin_a_algorithm_dijkstra_omp::TestTaskOpenMP task(task_data_omp);
  EXPECT_FALSE(task.Validation());
}

TEST(trubin_a_algorithm_dijkstra_omp, validation_fails_on_empty_inputs) {
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  trubin_a_algorithm_dijkstra_omp::TestTaskOpenMP task(task_data_omp);
  EXPECT_FALSE(task.Validation());
}

TEST(trubin_a_algorithm_dijkstra_omp, multi_component_graph) {
  RunDijkstraTest({1, 1, -1, -1, 3, 2, -1, -1}, 0, 4, {0, 1, -1, -1});
}

TEST(trubin_a_algorithm_dijkstra_omp, start_from_nonzero_vertex) {
  RunDijkstraTest({1, 3, -1, 2, 4, -1, 3, 2, -1, -1}, 2, 4, {-1, -1, 0, 2});
}

TEST(trubin_a_algorithm_dijkstra_omp, heavy_bucket_and_resize) {
  RunDijkstraTest({1, 20, -1, 2, 1, -1, -1}, 0, 3, {0, 20, 21});
}

TEST(trubin_a_algorithm_dijkstra_omp, heavy_bucket_triggered) { RunDijkstraTest({1, 10, -1, -1}, 0, 2, {0, 10}); }

TEST(trubin_a_algorithm_dijkstra_omp, bucket_resize_triggered) { RunDijkstraTest({1, 25, -1, -1}, 0, 2, {0, 25}); }
