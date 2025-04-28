#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/zinoviev_a_convex_hull_components/include/ops_tbb.hpp"

TEST(zinoviev_a_convex_hull_components_tbb, test_pipeline_run) {
  const int size = 5000;
  std::vector<int> input(size * size, 0);
  for (int i = 0; i < size; ++i) {
    input[i] = 1;
    input[((size - 1) * size) + i] = 1;
    input[i * size] = 1;
    input[(i * size) + (size - 1)] = 1;
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(size);
  task_data->inputs_count.emplace_back(size);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_tbb::Point[4]));
  task_data->outputs_count.emplace_back(4);

  auto task = std::make_shared<zinoviev_a_convex_hull_components_tbb::ConvexHullTBB>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 200;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  auto* output = reinterpret_cast<zinoviev_a_convex_hull_components_tbb::Point*>(task_data->outputs[0]);
  auto output_size = static_cast<size_t>(task_data->outputs_count[0]);
  std::vector<zinoviev_a_convex_hull_components_tbb::Point> actual(output, output + output_size);

  std::vector<zinoviev_a_convex_hull_components_tbb::Point> expected;
  expected.push_back({0, 0});
  expected.push_back({size - 1, 0});
  expected.push_back({size - 2, size - 1});
  expected.push_back({0, size - 1});

  std::set<zinoviev_a_convex_hull_components_tbb::Point> actual_set(actual.begin(), actual.end());
  std::set<zinoviev_a_convex_hull_components_tbb::Point> expected_set(expected.begin(), expected.end());

  ASSERT_EQ(actual.size(), expected.size());
  ASSERT_EQ(actual_set, expected_set);

  delete[] output;
}

TEST(zinoviev_a_convex_hull_components_tbb, test_task_run) {
  const int size = 5000;
  std::vector<int> input(size * size, 0);
  for (int i = 0; i < size; ++i) {
    input[i] = 1;
    input[((size - 1) * size) + i] = 1;
    input[i * size] = 1;
    input[(i * size) + (size - 1)] = 1;
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(size);
  task_data->inputs_count.emplace_back(size);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_tbb::Point[4]));
  task_data->outputs_count.emplace_back(4);

  auto task = std::make_shared<zinoviev_a_convex_hull_components_tbb::ConvexHullTBB>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 200;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  auto* output = reinterpret_cast<zinoviev_a_convex_hull_components_tbb::Point*>(task_data->outputs[0]);
  auto output_size = static_cast<size_t>(task_data->outputs_count[0]);
  std::vector<zinoviev_a_convex_hull_components_tbb::Point> actual(output, output + output_size);

  std::vector<zinoviev_a_convex_hull_components_tbb::Point> expected;
  expected.push_back({0, 0});
  expected.push_back({size - 1, 0});
  expected.push_back({size - 2, size - 1});
  expected.push_back({0, size - 1});

  std::set<zinoviev_a_convex_hull_components_tbb::Point> actual_set(actual.begin(), actual.end());
  std::set<zinoviev_a_convex_hull_components_tbb::Point> expected_set(expected.begin(), expected.end());

  ASSERT_EQ(actual.size(), expected.size());
  ASSERT_EQ(actual_set, expected_set);

  delete[] output;
}