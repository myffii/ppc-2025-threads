#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/kolodkin_g_multiplication_matrix_CRS/include/ops_omp.hpp"

namespace {
kolodkin_g_multiplication_matrix_omp::SparseMatrixCRS GenMatrix(
    unsigned int num_rows, unsigned int num_cols, unsigned int left_border_row, unsigned int right_border_row,
    unsigned int left_border_col, unsigned int right_border_col, int min_value, int max_value);
kolodkin_g_multiplication_matrix_omp::SparseMatrixCRS GenMatrix(
    unsigned int num_rows, unsigned int num_cols, unsigned int left_border_row, unsigned int right_border_row,
    unsigned int left_border_col, unsigned int right_border_col, int min_value, int max_value) {
  if (left_border_row > right_border_row || left_border_col > right_border_col || right_border_row > num_rows ||
      right_border_col > num_cols || min_value > max_value) {
    throw("ERROR!");
  }
  kolodkin_g_multiplication_matrix_omp::SparseMatrixCRS a((int)num_rows, (int)num_cols);
  for (unsigned int i = left_border_row; i < right_border_row; i++) {
    for (unsigned int j = left_border_col; j < right_border_col; j++) {
      a.AddValue((int)i, Complex(min_value + (rand() % max_value), min_value + (rand() % max_value)), (int)j);
    }
  }
  return a;
}
}  // namespace
TEST(kolodkin_g_multiplication_matrix__task_omp, test_pipeline_run) {
  srand(time(nullptr));
  kolodkin_g_multiplication_matrix_omp::SparseMatrixCRS a(400, 400);
  kolodkin_g_multiplication_matrix_omp::SparseMatrixCRS b(400, 400);
  std::vector<Complex> in = {};
  std::vector<Complex> in_a;
  std::vector<Complex> in_b;
  std::vector<Complex> out(a.numCols * b.numRows * 100, 0);

  a = ::GenMatrix(400, 400, 0, 150, 0, 150, -100, 100);
  b = ::GenMatrix(400, 400, 50, 140, 50, 150, -100, 100);
  in_a = kolodkin_g_multiplication_matrix_omp::ParseMatrixIntoVec(a);
  in_b = kolodkin_g_multiplication_matrix_omp::ParseMatrixIntoVec(b);
  in.reserve(in_a.size() + in_b.size());
  for (unsigned int i = 0; i < in_a.size(); i++) {
    in.emplace_back(in_a[i]);
  }
  for (unsigned int i = 0; i < in_b.size(); i++) {
    in.emplace_back(in_b[i]);
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential = std::make_shared<kolodkin_g_multiplication_matrix_omp::TestTaskOpenMP>(task_data_seq);

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
  kolodkin_g_multiplication_matrix_omp::SparseMatrixCRS res =
      kolodkin_g_multiplication_matrix_omp::ParseVectorIntoMatrix(out);
}

TEST(kolodkin_g_multiplication_matrix__task_omp, test_task_run) {
  srand(time(nullptr));
  kolodkin_g_multiplication_matrix_omp::SparseMatrixCRS a(400, 400);
  kolodkin_g_multiplication_matrix_omp::SparseMatrixCRS b(400, 400);
  std::vector<Complex> in = {};
  std::vector<Complex> in_a;
  std::vector<Complex> in_b;
  std::vector<Complex> out(a.numCols * b.numRows * 100, 0);

  a = ::GenMatrix(400, 400, 0, 150, 0, 150, -100, 100);
  b = ::GenMatrix(400, 400, 50, 140, 50, 150, -100, 100);
  in_a = kolodkin_g_multiplication_matrix_omp::ParseMatrixIntoVec(a);
  in_b = kolodkin_g_multiplication_matrix_omp::ParseMatrixIntoVec(b);
  in.reserve(in_a.size() + in_b.size());
  for (unsigned int i = 0; i < in_a.size(); i++) {
    in.emplace_back(in_a[i]);
  }
  for (unsigned int i = 0; i < in_b.size(); i++) {
    in.emplace_back(in_b[i]);
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential = std::make_shared<kolodkin_g_multiplication_matrix_omp::TestTaskOpenMP>(task_data_seq);

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
  kolodkin_g_multiplication_matrix_omp::SparseMatrixCRS res =
      kolodkin_g_multiplication_matrix_omp::ParseVectorIntoMatrix(out);
}
