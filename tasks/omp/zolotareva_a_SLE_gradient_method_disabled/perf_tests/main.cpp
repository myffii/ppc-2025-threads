#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/zolotareva_a_SLE_gradient_method/include/ops_seq.hpp"

void zolotareva_a_sle_gradient_method_omp::GenerateSle(std::vector<double> &a, std::vector<double> &b, int n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-100.0, 100.0);

  for (int i = 0; i < n; ++i) {
    b[i] = dist(gen);
    for (int j = i; j < n; ++j) {
      double value = dist(gen);
      a[(i * n) + j] = value;
      a[(j * n) + i] = value;
    }
  }

  for (int i = 0; i < n; ++i) {
    a[(i * n) + i] += n * 10.0;
  }
}

TEST(sequential_zolotareva_a_sle_gradient_method_omp, test_pipeline_run) {
  const int n = 1800;
  std::vector<double> a(n * n);
  std::vector<double> b(n);
  std::vector<double> x(n);
  zolotareva_a_sle_gradient_method_omp::GenerateSle(a, b, n);

  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_omp->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_omp->inputs_count.push_back(n * n);
  task_data_omp->inputs_count.push_back(n);
  task_data_omp->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_omp->outputs_count.push_back(x.size());

  auto test_task_omp = std::make_shared<zolotareva_a_sle_gradient_method_omp::TestTaskOpenMP>(task_data_omp);

  ASSERT_EQ(test_task_omp->ValidationImpl(), true);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();

  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  zolotareva_a_sle_gradient_method_omp::TestTaskOpenMP task(task_data_omp);
  ASSERT_EQ(task.ValidationImpl(), true);
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      sum += a[(i * n) + j] * x[j];
    }
    EXPECT_NEAR(sum, b[i], 1e-4);
  }
}

TEST(sequential_zolotareva_a_sle_gradient_method_omp, test_task_run) {
  const int n = 1800;
  std::vector<double> a(n * n);
  std::vector<double> b(n);
  std::vector<double> x(n);
  zolotareva_a_sle_gradient_method_omp::GenerateSle(a, b, n);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_omp->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_omp->inputs_count.push_back(n * n);
  task_data_omp->inputs_count.push_back(n);
  task_data_omp->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_omp->outputs_count.push_back(x.size());

  auto test_task_omp = std::make_shared<zolotareva_a_sle_gradient_method_omp::TestTaskOpenMP>(task_data_omp);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  zolotareva_a_sle_gradient_method_omp::TestTaskOpenMP task(task_data_omp);
  ASSERT_EQ(task.ValidationImpl(), true);
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      sum += a[(i * n) + j] * x[j];
    }
    EXPECT_NEAR(sum, b[i], 1e-4);
  }
}
