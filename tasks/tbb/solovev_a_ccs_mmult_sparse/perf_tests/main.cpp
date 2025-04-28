#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/solovev_a_ccs_mmult_sparse/include/ccs_mmult_sparse_tbb.hpp"

namespace {
std::complex<double> GenerateRandomComplex(double min, double max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min, max);
  return {dis(gen), dis(gen)};
}

bool AreComplexNumbersApproxEqual(const std::complex<double>& c1, const std::complex<double>& c2,
                                  double tolerance = 1e-6) {
  return std::abs(c1.real() - c2.real()) < tolerance && std::abs(c1.imag() - c2.imag()) < tolerance;
}
}  // namespace

TEST(solovev_a_ccs_mmult_sparse_tbb, test_pipeline_run) {
  int rows = 2000000;
  int cols = 2000000;
  solovev_a_matrix_tbb::MatrixInCcsSparse m1(rows, cols);
  solovev_a_matrix_tbb::MatrixInCcsSparse m2(rows, 1);
  solovev_a_matrix_tbb::MatrixInCcsSparse m3(rows, 1);

  for (int i = 0; i <= cols; i++) {
    m1.col_p.push_back(i);
  }

  for (int i = 0; i < m1.col_p[cols]; i++) {
    m1.val.emplace_back(GenerateRandomComplex(-10.0, 10.0));
    m1.row.push_back(i);
  }

  m2.col_p = {0, rows};
  for (int i = 0; i < rows; i++) {
    m2.val.emplace_back(GenerateRandomComplex(-10.0, 10.0));
    m2.row.push_back(i);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m1));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m2));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(&m3));

  auto test_task_tbb = std::make_shared<solovev_a_matrix_tbb::TBBMatMultCcs>(task_data_tbb);

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

  for (size_t i = 0; i < m3.val.size(); i++) {
    bool approx_equal = AreComplexNumbersApproxEqual(m3.val[i], m1.val[i] * m2.val[i]);
    ASSERT_TRUE(approx_equal);
  }
}

TEST(solovev_a_ccs_mmult_sparse_tbb, test_task_run) {
  int rows = 2000000;
  int cols = 2000000;
  solovev_a_matrix_tbb ::MatrixInCcsSparse m1(rows, cols);
  solovev_a_matrix_tbb::MatrixInCcsSparse m2(rows, 1);
  solovev_a_matrix_tbb::MatrixInCcsSparse m3(rows, 1);

  for (int i = 0; i <= cols; i++) {
    m1.col_p.push_back(i);
  }

  for (int i = 0; i < m1.col_p[cols]; i++) {
    m1.val.emplace_back(GenerateRandomComplex(-10.0, 10.0));
    m1.row.push_back(i);
  }

  m2.col_p = {0, rows};
  for (int i = 0; i < rows; i++) {
    m2.val.emplace_back(GenerateRandomComplex(-10.0, 10.0));
    m2.row.push_back(i);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m1));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m2));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(&m3));

  auto test_task_omp = std::make_shared<solovev_a_matrix_tbb::TBBMatMultCcs>(task_data_tbb);

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

  for (size_t i = 0; i < m3.val.size(); i++) {
    bool approx_equal = AreComplexNumbersApproxEqual(m3.val[i], m1.val[i] * m2.val[i]);
    ASSERT_TRUE(approx_equal);
  }
}