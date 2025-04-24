#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/kolokolova_d_integral_simpson_method/include/ops_omp.hpp"

TEST(kolokolova_d_integral_simpson_method_omp, test_pipeline_run) {
  auto func = [](std::vector<double> vec) {
    return (vec[2] * vec[2] * vec[2] * vec[1] * vec[1] / 10) + (4 * vec[0] * vec[0]) - (10 * vec[2]);
  };
  std::vector<int> step = {130, 130, 130};
  std::vector<int> bord = {1, 11, 2, 10, 0, 10};
  double func_result = 0.0;

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_omp->inputs_count.emplace_back(step.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_omp->inputs_count.emplace_back(bord.size());

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_omp->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_omp = std::make_shared<kolokolova_d_integral_simpson_method_omp::TestTaskOpenMP>(task_data_omp, func);

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

  double ans = 927300.25;
  double error = 0.1;
  ASSERT_NEAR(func_result, ans, error);
}

TEST(kolokolova_d_integral_simpson_method_omp, test_task_run) {
  auto func = [](std::vector<double> vec) {
    return (vec[2] * vec[2] * vec[2] * vec[1] * vec[1] / 10) + (4 * vec[0] * vec[0]) - (10 * vec[2]);
  };
  std::vector<int> step = {130, 130, 130};
  std::vector<int> bord = {1, 11, 2, 10, 0, 10};
  double func_result = 0.0;

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_omp->inputs_count.emplace_back(step.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_omp->inputs_count.emplace_back(bord.size());

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_omp->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_omp = std::make_shared<kolokolova_d_integral_simpson_method_omp::TestTaskOpenMP>(task_data_omp, func);

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
  double ans = 927300.25;
  double error = 0.1;
  ASSERT_NEAR(func_result, ans, error);
}
