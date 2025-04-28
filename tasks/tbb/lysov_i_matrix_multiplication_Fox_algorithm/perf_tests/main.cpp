#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_tbb.hpp"

namespace lysov_i_matrix_multiplication_fox_algorithm_tbb {
std::vector<double> GetRandomMatrix(size_t size, int min_gen_value, int max_gen_value) {
  std::vector<double> matrix(size * size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min_gen_value, max_gen_value);

  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      matrix[(i * size) + j] = dist(gen);
    }
  }
  return matrix;
}
void TrivialMatrixMultiplication(const std::vector<double> &matrix_a, const std::vector<double> &matrix_b,
                                 std::vector<double> &result_matrix, size_t matrix_size) {
  for (size_t row = 0; row < matrix_size; ++row) {
    for (size_t col = 0; col < matrix_size; ++col) {
      result_matrix[(row * matrix_size) + col] = 0.0;
      for (size_t k = 0; k < matrix_size; ++k) {
        result_matrix[(row * matrix_size) + col] +=
            matrix_a[(row * matrix_size) + k] * matrix_b[(k * matrix_size) + col];
      }
      result_matrix[(row * matrix_size) + col] = round(result_matrix[(row * matrix_size) + col] * 10000) / 10000;
    }
  }
}
}  // namespace lysov_i_matrix_multiplication_fox_algorithm_tbb

TEST(lysov_i_matrix_multiplication_Fox_algorithm_omp, test_pipeline_run) {
  // Create data
  size_t n = 800;
  size_t block_size = 30;
  int min_gen_value = -1e3;
  int max_gen_value = 1e3;
  std::vector<double> a =
      lysov_i_matrix_multiplication_fox_algorithm_tbb::GetRandomMatrix(n, min_gen_value, max_gen_value);
  std::vector<double> b =
      lysov_i_matrix_multiplication_fox_algorithm_tbb::GetRandomMatrix(n, min_gen_value, max_gen_value);
  std::vector<double> c(n * n, 0);
  std::vector<double> c_expected(n * n, 0);
  lysov_i_matrix_multiplication_fox_algorithm_tbb::TrivialMatrixMultiplication(a, b, c_expected, n);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_tbb->inputs_count.emplace_back(n * n);
  task_data_tbb->inputs_count.emplace_back(n * n);
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->outputs_count.emplace_back(n * n);
  ;
  lysov_i_matrix_multiplication_fox_algorithm_tbb::TestTaskTBB matrix_multiplication(task_data_tbb);

  // Create Task
  auto test_task_sequential =
      std::make_shared<lysov_i_matrix_multiplication_fox_algorithm_tbb::TestTaskTBB>(task_data_tbb);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (size_t i = 0; i < n * n; i++) {
    EXPECT_NEAR(c_expected[i], c[i], 1e-3);
  }
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_omp, test_task_run) {
  size_t n = 800;
  size_t block_size = 30;
  int min_gen_value = -1e3;
  int max_gen_value = 1e3;
  std::vector<double> a =
      lysov_i_matrix_multiplication_fox_algorithm_tbb::GetRandomMatrix(n, min_gen_value, max_gen_value);
  std::vector<double> b =
      lysov_i_matrix_multiplication_fox_algorithm_tbb::GetRandomMatrix(n, min_gen_value, max_gen_value);
  std::vector<double> c(n * n, 0);
  std::vector<double> c_expected(n * n, 0);
  lysov_i_matrix_multiplication_fox_algorithm_tbb::TrivialMatrixMultiplication(a, b, c_expected, n);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_tbb->inputs_count.emplace_back(n * n);
  task_data_tbb->inputs_count.emplace_back(n * n);
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->outputs_count.emplace_back(n * n);

  // Create Task
  auto test_task_sequential =
      std::make_shared<lysov_i_matrix_multiplication_fox_algorithm_tbb::TestTaskTBB>(task_data_tbb);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (size_t i = 0; i < n * n; i++) {
    EXPECT_NEAR(c_expected[i], c[i], 1e-3);
  }
}