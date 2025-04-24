#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/malyshev_v_radix_sort/include/ops_omp.hpp"

TEST(malyshev_v_radix_sort_omp, test_pipeline) {
  const size_t k_size = 1e6;
  std::vector<double> input(k_size);
  std::vector<double> output(k_size);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1e6, 1e6);
  for (auto& val : input) {
    val = dist(gen);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<malyshev_v_radix_sort_omp::RadixSortDoubleOMP>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  perf_attr->current_timer = [&] { return omp_get_wtime(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_TRUE(std::ranges::is_sorted(output));
}

TEST(malyshev_v_radix_sort_omp, test_task_run) {
  const size_t k_size = 1e6;
  std::vector<double> input(k_size);
  std::vector<double> output(k_size);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1e6, 1e6);
  for (auto& val : input) {
    val = dist(gen);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<malyshev_v_radix_sort_omp::RadixSortDoubleOMP>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  perf_attr->current_timer = [&] { return omp_get_wtime(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_TRUE(std::ranges::is_sorted(output));
}