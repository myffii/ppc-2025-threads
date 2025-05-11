#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/opolin_d_radix_sort_batcher_merge/include/ops_tbb.hpp"

namespace opolin_d_radix_batcher_sort_tbb {
namespace {
void GenDataRadixSort(size_t size, std::vector<int> &vec, std::vector<int> &expected) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(-1000, 1000);
  vec.clear();
  expected.clear();
  vec.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    vec.push_back(dis(gen));
  }
  expected = vec;
  std::ranges::sort(expected);
}
}  // namespace
}  // namespace opolin_d_radix_batcher_sort_tbb

TEST(opolin_d_radix_batcher_sort_tbb, test_pipeline_run) {
  int size = 500000;
  std::vector<int> input;
  std::vector<int> expected;
  opolin_d_radix_batcher_sort_tbb::GenDataRadixSort(size, input, expected);
  std::vector<int> out(size, 0);
  // Create TaskData
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_tbb->inputs_count.emplace_back(out.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_tbb = std::make_shared<opolin_d_radix_batcher_sort_tbb::RadixBatcherSortTaskTbb>(task_data_tbb);
  ASSERT_EQ(test_task_tbb->Validation(), true);
  test_task_tbb->PreProcessing();
  test_task_tbb->Run();
  test_task_tbb->PostProcessing();
  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(expected, out);
}

TEST(opolin_d_radix_batcher_sort_tbb, test_task_run) {
  int size = 500000;
  std::vector<int> input;
  std::vector<int> expected;
  opolin_d_radix_batcher_sort_tbb::GenDataRadixSort(size, input, expected);
  std::vector<int> out(size, 0);
  // Create TaskData
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_tbb->inputs_count.emplace_back(out.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_tbb = std::make_shared<opolin_d_radix_batcher_sort_tbb::RadixBatcherSortTaskTbb>(task_data_tbb);
  ASSERT_EQ(test_task_tbb->Validation(), true);
  test_task_tbb->PreProcessing();
  test_task_tbb->Run();
  test_task_tbb->PostProcessing();
  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(expected, out);
}