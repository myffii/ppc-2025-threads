#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/borisov_s_strassen/include/ops_stl.hpp"

namespace {

std::vector<double> MultiplyNaiveDouble(const std::vector<double>& a, const std::vector<double>& b, int rows_a,
                                        int cols_a, int cols_b) {
  std::vector<double> c(rows_a * cols_b, 0.0);
  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      double sum = 0.0;
      for (int k = 0; k < cols_a; ++k) {
        sum += a[(i * cols_a) + k] * b[(k * cols_b) + j];
      }
      c[(i * cols_b) + j] = sum;
    }
  }
  return c;
}

void GenerateRandomMatrix(int rows, int cols, std::vector<double>& matrix) {
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<double> dist(-50.0, 50.0);
  matrix.resize(rows * cols);
  for (auto& value : matrix) {
    value = dist(rng);
  }
}

}  // namespace

TEST(borisov_s_strassen_perf_stl, test_pipeline_run) {
  constexpr int kRowsA = 1024;
  constexpr int kColsA = 512;
  constexpr int kRowsB = 512;
  constexpr int kColsB = 1024;

  std::vector<double> a;
  std::vector<double> b;
  GenerateRandomMatrix(kRowsA, kColsA, a);
  GenerateRandomMatrix(kRowsB, kColsB, b);

  std::vector<double> in_data = {static_cast<double>(kRowsA), static_cast<double>(kColsA), static_cast<double>(kRowsB),
                                 static_cast<double>(kColsB)};
  in_data.insert(in_data.end(), a.begin(), a.end());
  in_data.insert(in_data.end(), b.begin(), b.end());

  size_t output_count = 2 + (kRowsA * kColsB);
  std::vector<double> out(output_count, 0.0);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data_stl->inputs_count.emplace_back(in_data.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  auto test_task_parallel = std::make_shared<borisov_s_strassen_stl::ParallelStrassenStl>(task_data_stl);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_parallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<double> c_expected = MultiplyNaiveDouble(a, b, kRowsA, kColsA, kColsB);
  std::vector<double> c_result(out.begin() + 2, out.end());

  ASSERT_EQ(c_expected.size(), c_result.size());
  for (std::size_t i = 0; i < c_expected.size(); ++i) {
    ASSERT_NEAR(c_expected[i], c_result[i], 1e-8);
  }
}

TEST(borisov_s_strassen_perf_stl, test_task_run) {
  constexpr int kRowsA = 1024;
  constexpr int kColsA = 512;
  constexpr int kRowsB = 512;
  constexpr int kColsB = 1024;

  std::vector<double> a;
  std::vector<double> b;
  GenerateRandomMatrix(kRowsA, kColsA, a);
  GenerateRandomMatrix(kRowsB, kColsB, b);

  std::vector<double> in_data = {static_cast<double>(kRowsA), static_cast<double>(kColsA), static_cast<double>(kRowsB),
                                 static_cast<double>(kColsB)};
  in_data.insert(in_data.end(), a.begin(), a.end());
  in_data.insert(in_data.end(), b.begin(), b.end());

  size_t output_count = 2 + (kRowsA * kColsB);
  std::vector<double> out(output_count, 0.0);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data_stl->inputs_count.emplace_back(in_data.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  auto test_task_parallel = std::make_shared<borisov_s_strassen_stl::ParallelStrassenStl>(task_data_stl);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<double> c_expected = MultiplyNaiveDouble(a, b, kRowsA, kColsA, kColsB);
  std::vector<double> c_result(out.begin() + 2, out.end());

  ASSERT_EQ(c_expected.size(), c_result.size());
  for (std::size_t i = 0; i < c_expected.size(); ++i) {
    ASSERT_NEAR(c_expected[i], c_result[i], 1e-8);
  }
}