#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/gusev_n_sorting_int_simple_merging/include/ops_tbb.hpp"

namespace {
struct TestData {
  std::vector<int> in;
  std::vector<int> out;
  ppc::core::TaskDataPtr task_data;
};

TestData GenerateTestData(int count) {
  TestData data;
  data.in.resize(count);
  data.out.resize(count);

  std::ranges::generate(data.in, []() { return (std::rand() % 20000) - 10000; });

  data.task_data = std::make_shared<ppc::core::TaskData>();
  data.task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.in.data()));
  data.task_data->inputs_count.emplace_back(data.in.size());
  data.task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(data.out.data()));
  data.task_data->outputs_count.emplace_back(data.out.size());

  return data;
}
}  // namespace

TEST(gusev_n_sorting_int_simple_merging_tbb, test_pipeline_run) {
  constexpr int kCount = 5000000;
  auto test_data = GenerateTestData(kCount);

  auto test_task =
      std::make_shared<gusev_n_sorting_int_simple_merging_tbb::SortingIntSimpleMergingTBB>(test_data.task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  std::vector<int> expected = test_data.in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, test_data.out);
}

TEST(gusev_n_sorting_int_simple_merging_tbb, test_task_run) {
  constexpr int kCount = 5000000;
  auto test_data = GenerateTestData(kCount);

  auto test_task =
      std::make_shared<gusev_n_sorting_int_simple_merging_tbb::SortingIntSimpleMergingTBB>(test_data.task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  std::vector<int> expected = test_data.in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, test_data.out);
}
