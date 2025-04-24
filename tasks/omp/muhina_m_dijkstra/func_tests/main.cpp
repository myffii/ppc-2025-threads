#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/muhina_m_dijkstra/include/ops_omp.hpp"

namespace {
std::vector<int> DijkstraSequential(const std::vector<std::vector<std::pair<size_t, int>>>& adj_list,
                                    size_t start_vertex) {
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

std::vector<std::vector<std::pair<size_t, int>>> GenerateRandomGraph(size_t min_vertices = 5, size_t max_vertices = 20,
                                                                     int min_weight = 1, int max_weight = 100,
                                                                     size_t min_edges_per_vertex = 1,
                                                                     size_t max_edges_per_vertex = 3) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> vertices_dist(min_vertices, max_vertices);
  std::uniform_int_distribution<int> weight_dist(min_weight, max_weight);
  std::uniform_int_distribution<size_t> edges_dist(min_edges_per_vertex, max_edges_per_vertex);

  const size_t num_vertices = vertices_dist(gen);
  std::vector<std::vector<std::pair<size_t, int>>> adj_list(num_vertices);

  for (size_t u = 0; u < num_vertices; ++u) {
    size_t num_edges = edges_dist(gen);
    for (size_t j = 0; j < num_edges; ++j) {
      size_t v = std::uniform_int_distribution<size_t>(0, num_vertices - 1)(gen);
      if (v == u) {
        v = (v + 1) % num_vertices;
      }
      int weight = weight_dist(gen);
      adj_list[u].emplace_back(v, weight);
    }
  }

  return adj_list;
}
}  // namespace

TEST(muhina_m_dijkstra_omp, test_dijkstra_small_graph) {
  constexpr size_t kNumVertices = 5;
  std::vector<std::vector<std::pair<size_t, int>>> adj_list(kNumVertices);
  adj_list[0].emplace_back(1, 4);
  adj_list[0].emplace_back(2, 2);
  adj_list[1].emplace_back(2, 5);
  adj_list[1].emplace_back(3, 10);
  adj_list[2].emplace_back(3, 3);
  adj_list[3].emplace_back(4, 4);
  adj_list[2].emplace_back(4, 1);

  size_t start_vertex = 0;

  std::vector<int> distances(kNumVertices, INT_MAX);
  std::vector<int> graph_data;
  for (const auto& vertex_edges : adj_list) {
    for (const auto& edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_omp->inputs_count.emplace_back(graph_data.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_omp->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_omp->outputs_count.emplace_back(kNumVertices);

  muhina_m_dijkstra_omp::TestTaskOpenMP test_task_open_mp(task_data_omp);
  ASSERT_TRUE(test_task_open_mp.Validation());
  test_task_open_mp.PreProcessing();
  ASSERT_TRUE(test_task_open_mp.Run());
  test_task_open_mp.PostProcessing();

  std::vector<int> expected_distances = {0, 4, 2, 5, 3};
  for (size_t i = 0; i < kNumVertices; ++i) {
    EXPECT_EQ(distances[i], expected_distances[i]);
  }
}

TEST(muhina_m_dijkstra_omp, test_dijkstra_validation_failure) {
  std::vector<int> graph_data;
  size_t start_vertex = 0;
  size_t num_vertices = 0;
  std::vector<int> distances(num_vertices, INT_MAX);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_omp->inputs_count.emplace_back(graph_data.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_omp->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_omp->outputs_count.emplace_back(num_vertices);

  muhina_m_dijkstra_omp::TestTaskOpenMP test_task_open_mp(task_data_omp);

  ASSERT_FALSE(test_task_open_mp.Validation());
}

TEST(muhina_m_dijkstra_omp, test_dijkstra_small_graph_non_zero_start) {
  constexpr size_t kNumVertices = 5;
  std::vector<std::vector<std::pair<size_t, int>>> adj_list(kNumVertices);
  adj_list[0].emplace_back(1, 4);
  adj_list[0].emplace_back(2, 2);
  adj_list[1].emplace_back(2, 5);
  adj_list[1].emplace_back(3, 10);
  adj_list[2].emplace_back(3, 3);
  adj_list[3].emplace_back(4, 4);
  adj_list[2].emplace_back(4, 1);
  adj_list[2].emplace_back(0, 2);
  adj_list[2].emplace_back(1, 5);

  size_t start_vertex = 2;

  std::vector<int> distances(kNumVertices, INT_MAX);
  std::vector<int> graph_data;
  for (const auto& vertex_edges : adj_list) {
    for (const auto& edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_omp->inputs_count.emplace_back(graph_data.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_omp->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_omp->outputs_count.emplace_back(kNumVertices);

  muhina_m_dijkstra_omp::TestTaskOpenMP test_task_open_mp(task_data_omp);
  ASSERT_TRUE(test_task_open_mp.Validation());
  test_task_open_mp.PreProcessing();
  ASSERT_TRUE(test_task_open_mp.Run());
  test_task_open_mp.PostProcessing();

  std::vector<int> expected_distances = {2, 5, 0, 3, 1};
  for (size_t i = 0; i < kNumVertices; ++i) {
    EXPECT_EQ(distances[i], expected_distances[i]);
  }
}

TEST(muhina_m_dijkstra_omp, test_negative_weight) {
  constexpr size_t kNumVertices = 5;
  std::vector<std::vector<std::pair<size_t, int>>> adj_list(kNumVertices);
  adj_list[0].emplace_back(1, -4);
  adj_list[0].emplace_back(2, 2);
  adj_list[1].emplace_back(2, 5);
  adj_list[1].emplace_back(3, 10);
  adj_list[2].emplace_back(3, -3);
  adj_list[3].emplace_back(4, 4);
  adj_list[2].emplace_back(4, 1);

  size_t start_vertex = 0;

  std::vector<int> distances(kNumVertices, INT_MAX);
  std::vector<int> graph_data;
  for (const auto& vertex_edges : adj_list) {
    for (const auto& edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_omp->inputs_count.emplace_back(graph_data.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_omp->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_omp->outputs_count.emplace_back(kNumVertices);

  muhina_m_dijkstra_omp::TestTaskOpenMP test_task_open_mp(task_data_omp);
  ASSERT_TRUE(test_task_open_mp.Validation());
  test_task_open_mp.PreProcessing();
  ASSERT_FALSE(test_task_open_mp.Run());
  test_task_open_mp.PostProcessing();
}

TEST(muhina_m_dijkstra_omp, test_dijkstra_random_graph) {
  auto adj_list = GenerateRandomGraph();
  const size_t num_vertices = adj_list.size();
  size_t start_vertex = 0;

  auto expected_distances = DijkstraSequential(adj_list, start_vertex);

  std::vector<int> distances(num_vertices, INT_MAX);
  std::vector<int> graph_data;
  for (const auto& vertex_edges : adj_list) {
    for (const auto& edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_omp->inputs_count.emplace_back(graph_data.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_omp->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_omp->outputs_count.emplace_back(num_vertices);

  muhina_m_dijkstra_omp::TestTaskOpenMP test_task_open_mp(task_data_omp);
  ASSERT_TRUE(test_task_open_mp.Validation());
  test_task_open_mp.PreProcessing();
  ASSERT_TRUE(test_task_open_mp.Run());
  test_task_open_mp.PostProcessing();

  for (size_t i = 0; i < num_vertices; ++i) {
    EXPECT_EQ(distances[i], expected_distances[i]);
  }
}

TEST(muhina_m_dijkstra_omp, single_vertex_graph) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {{}};
  size_t start_vertex = 0;

  std::vector<int> graph_data = {-1};
  std::vector<int> distances(1, INT_MAX);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_omp->inputs_count.emplace_back(graph_data.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_omp->inputs_count.emplace_back(sizeof(start_vertex));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_omp->outputs_count.emplace_back(1);

  muhina_m_dijkstra_omp::TestTaskOpenMP test_task_open_mp(task_data_omp);
  ASSERT_TRUE(test_task_open_mp.Validation());
  test_task_open_mp.PreProcessing();
  ASSERT_TRUE(test_task_open_mp.Run());
  test_task_open_mp.PostProcessing();

  EXPECT_EQ(distances[0], 0);
}

TEST(muhina_m_dijkstra_omp, two_connected_components) {
  constexpr size_t kNumVertices = 6;
  std::vector<std::vector<std::pair<size_t, int>>> adj_list(kNumVertices);

  adj_list[0].emplace_back(1, 2);
  adj_list[0].emplace_back(2, 4);
  adj_list[1].emplace_back(0, 2);
  adj_list[1].emplace_back(2, 1);
  adj_list[2].emplace_back(0, 4);
  adj_list[2].emplace_back(1, 1);

  adj_list[3].emplace_back(4, 3);
  adj_list[3].emplace_back(5, 7);
  adj_list[4].emplace_back(3, 3);
  adj_list[4].emplace_back(5, 2);
  adj_list[5].emplace_back(3, 7);
  adj_list[5].emplace_back(4, 2);

  size_t start_vertex = 0;

  std::vector<int> distances(kNumVertices, INT_MAX);
  std::vector<int> graph_data;
  for (const auto& vertex_edges : adj_list) {
    for (const auto& edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_omp->inputs_count.emplace_back(graph_data.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_omp->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_omp->outputs_count.emplace_back(kNumVertices);

  muhina_m_dijkstra_omp::TestTaskOpenMP test_task_open_mp(task_data_omp);
  ASSERT_TRUE(test_task_open_mp.Validation());
  test_task_open_mp.PreProcessing();
  ASSERT_TRUE(test_task_open_mp.Run());
  test_task_open_mp.PostProcessing();

  std::vector<int> expected_distances = {0, 2, 3, INT_MAX, INT_MAX, INT_MAX};
  for (size_t i = 0; i < kNumVertices; ++i) {
    EXPECT_EQ(distances[i], expected_distances[i]);
  }
}