#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/khovansky_d_double_radix_batcher/include/ops_omp.hpp"

TEST(khovansky_d_double_radix_batcher_omp, test_pipeline_run) {
  constexpr int kCount = 1000000;

  // Create data
  std::vector<double> in(kCount);
  std::vector<double> exp_out(kCount);

  for (size_t i = 0; i < kCount; ++i) {
    in[i] = static_cast<double>(kCount - i);
    exp_out[i] = static_cast<double>(i + 1);
  }

  std::vector<double> out(kCount);

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_omp = std::make_shared<khovansky_d_double_radix_batcher_omp::RadixOMP>(task_data_omp);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(exp_out, out);
}

TEST(khovansky_d_double_radix_batcher_omp, test_task_run) {
  constexpr int kCount = 1000000;

  // Create data
  std::vector<double> in(kCount);
  std::vector<double> exp_out(kCount);

  for (size_t i = 0; i < kCount; ++i) {
    in[i] = static_cast<double>(kCount - i);
    exp_out[i] = static_cast<double>(i + 1);
  }

  std::vector<double> out(kCount);

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_omp = std::make_shared<khovansky_d_double_radix_batcher_omp::RadixOMP>(task_data_omp);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(exp_out, out);
}