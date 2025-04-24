#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/moiseev_a_mult_mat/include/ops_tbb.hpp"

namespace {

std::vector<double> GenerateRandomMatrix(size_t rows, size_t cols) {
  std::vector<double> matrix(rows * cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-100.0, 100.0);

  for (auto &val : matrix) {
    val = dist(gen);
  }
  return matrix;
}

void NaiveMultiply(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < n; ++k) {
        sum += a[(i * n) + k] * b[(k * n) + j];
      }
      c[(i * n) + j] = sum;
    }
  }
}

}  // namespace

TEST(moiseev_a_mult_mat_tbb, test_pipeline_run) {
  constexpr int kCount = 500;

  auto matrix_a = GenerateRandomMatrix(kCount, kCount);
  auto matrix_b = GenerateRandomMatrix(kCount, kCount);
  std::vector<double> matrix_c(kCount * kCount, 0.0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_tbb->inputs_count.emplace_back(matrix_a.size());

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_tbb->inputs_count.emplace_back(matrix_b.size());

  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  task_data_tbb->outputs_count.emplace_back(matrix_c.size());

  auto test_task_tbb = std::make_shared<moiseev_a_mult_mat_tbb::MultMatTBB>(task_data_tbb);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<double> expected_matrix(kCount * kCount, 0.0);
  NaiveMultiply(matrix_a, matrix_b, expected_matrix, kCount);

  const double epsilon = 1e-6;
  for (size_t i = 0; i < matrix_c.size(); ++i) {
    EXPECT_NEAR(matrix_c[i], expected_matrix[i], epsilon);
  }
}

TEST(moiseev_a_mult_mat_tbb, test_task_run) {
  constexpr int kCount = 500;

  auto matrix_a = GenerateRandomMatrix(kCount, kCount);
  auto matrix_b = GenerateRandomMatrix(kCount, kCount);
  std::vector<double> matrix_c(kCount * kCount, 0.0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_tbb->inputs_count.emplace_back(matrix_a.size());

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_tbb->inputs_count.emplace_back(matrix_b.size());

  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  task_data_tbb->outputs_count.emplace_back(matrix_c.size());

  auto test_task_tbb = std::make_shared<moiseev_a_mult_mat_tbb::MultMatTBB>(task_data_tbb);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<double> expected_matrix(kCount * kCount, 0.0);
  NaiveMultiply(matrix_a, matrix_b, expected_matrix, kCount);

  const double epsilon = 1e-6;
  for (size_t i = 0; i < matrix_c.size(); ++i) {
    EXPECT_NEAR(matrix_c[i], expected_matrix[i], epsilon);
  }
}
