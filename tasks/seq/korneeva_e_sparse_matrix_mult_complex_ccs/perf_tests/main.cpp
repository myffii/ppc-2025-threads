#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/korneeva_e_sparse_matrix_mult_complex_ccs/include/ops_seq.hpp"

namespace korneeva_e_ccs = korneeva_e_sparse_matrix_mult_complex_ccs_seq;

namespace {
korneeva_e_ccs::SparseMatrixCCS CreateRandomMatrix(int rows, int cols, int max_nnz) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  korneeva_e_ccs::SparseMatrixCCS matrix(rows, cols, 0);
  std::uniform_real_distribution<> dis(-10.0, 10.0);
  std::uniform_int_distribution<> row_dis(0, rows - 1);

  std::vector<std::vector<korneeva_e_ccs::Complex>> temp(rows, std::vector<korneeva_e_ccs::Complex>(cols, {0.0, 0.0}));
  int nnz = 0;
  while (nnz < max_nnz && nnz < rows * cols) {
    int r = row_dis(gen);
    int c = row_dis(gen) % cols;
    if (temp[r][c] == korneeva_e_ccs::Complex(0.0, 0.0)) {
      temp[r][c] = korneeva_e_ccs::Complex(dis(gen), dis(gen));
      nnz++;
    }
  }

  matrix.nnz = nnz;
  matrix.values.reserve(nnz);
  matrix.row_indices.reserve(nnz);
  matrix.col_offsets.resize(cols + 1, 0);

  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      if (temp[i][j] != korneeva_e_ccs::Complex(0.0, 0.0)) {
        matrix.values.push_back(temp[i][j]);
        matrix.row_indices.push_back(i);
      }
    }
    matrix.col_offsets[j + 1] = static_cast<int>(matrix.values.size());
  }
  return matrix;
}
}  // namespace

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_pipeline_run) {
  constexpr int kRowsCols = 500;
  constexpr int kMaxNnzMatrix = 500;
  constexpr int kMaxNnzMatrix2 = 500;

  korneeva_e_ccs::SparseMatrixCCS matrix1 = CreateRandomMatrix(kRowsCols, kRowsCols, kMaxNnzMatrix);
  korneeva_e_ccs::SparseMatrixCCS matrix2 = CreateRandomMatrix(kRowsCols, kRowsCols, kMaxNnzMatrix2);
  korneeva_e_ccs::SparseMatrixCCS result;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  auto test_task_sequential = std::make_shared<korneeva_e_ccs::SparseMatrixMultComplexCCS>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(result.rows, kRowsCols);
  ASSERT_EQ(result.cols, kRowsCols);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_task_run) {
  constexpr int kRowsCols = 500;
  constexpr int kMaxNnzMatrix = 500;
  constexpr int kMaxNnzMatrix2 = 500;

  korneeva_e_ccs::SparseMatrixCCS matrix1 = CreateRandomMatrix(kRowsCols, kRowsCols, kMaxNnzMatrix);
  korneeva_e_ccs::SparseMatrixCCS matrix2 = CreateRandomMatrix(kRowsCols, kRowsCols, kMaxNnzMatrix2);
  korneeva_e_ccs::SparseMatrixCCS result;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  auto test_task_sequential = std::make_shared<korneeva_e_ccs::SparseMatrixMultComplexCCS>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(result.rows, kRowsCols);
  ASSERT_EQ(result.cols, kRowsCols);
}