#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/korneeva_e_sparse_matrix_mult_complex_ccs/include/ops_tbb.hpp"

namespace korneeva_e_tbb = korneeva_e_sparse_matrix_mult_complex_ccs_tbb;

namespace {
void RunTask(korneeva_e_tbb::SparseMatrixCCS& m1, korneeva_e_tbb::SparseMatrixCCS& m2,
             korneeva_e_tbb::SparseMatrixCCS& result) {
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m1));
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m2));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));

  korneeva_e_tbb::SparseMatrixMultComplexCCS task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

void ExpectMatrixValuesEq(const korneeva_e_tbb::SparseMatrixCCS& result,
                          const korneeva_e_tbb::SparseMatrixCCS& expected, double epsilon) {
  ASSERT_EQ(result.values.size(), expected.values.size());
  for (size_t i = 0; i < result.values.size(); i++) {
    EXPECT_NEAR(std::abs(result.values[i] - expected.values[i]), 0.0, epsilon);
  }
}

void ExpectMatrixEq(const korneeva_e_tbb::SparseMatrixCCS& result, const korneeva_e_tbb::SparseMatrixCCS& expected,
                    double epsilon = 1e-6) {
  EXPECT_EQ(result.rows, expected.rows);
  EXPECT_EQ(result.cols, expected.cols);
  EXPECT_EQ(result.nnz, expected.nnz);
  EXPECT_EQ(result.col_offsets, expected.col_offsets);
  EXPECT_EQ(result.row_indices, expected.row_indices);
  ExpectMatrixValuesEq(result, expected, epsilon);
}

korneeva_e_tbb::SparseMatrixCCS CreateRandomMatrix(int rows, int cols, int max_nnz) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  korneeva_e_tbb::SparseMatrixCCS matrix(rows, cols, 0);
  std::uniform_real_distribution<> dis(-10.0, 10.0);
  std::uniform_int_distribution<> row_dis(0, rows - 1);

  std::vector<std::vector<korneeva_e_tbb::Complex>> temp(rows, std::vector<korneeva_e_tbb::Complex>(cols, {0.0, 0.0}));
  int nnz = 0;
  while (nnz < max_nnz && nnz < rows * cols) {
    int r = row_dis(gen);
    int c = row_dis(gen) % cols;
    if (temp[r][c] == korneeva_e_tbb::Complex(0.0, 0.0)) {
      temp[r][c] = korneeva_e_tbb::Complex(dis(gen), dis(gen));
      nnz++;
    }
  }

  matrix.nnz = nnz;
  matrix.values.reserve(nnz);
  matrix.row_indices.reserve(nnz);
  matrix.col_offsets.resize(cols + 1, 0);

  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      if (temp[i][j] != korneeva_e_tbb::Complex(0.0, 0.0)) {
        matrix.values.push_back(temp[i][j]);
        matrix.row_indices.push_back(i);
      }
    }
    matrix.col_offsets[j + 1] = static_cast<int>(matrix.values.size());
  }
  return matrix;
}

korneeva_e_tbb::SparseMatrixCCS CreateCcsFromDense(const std::vector<std::vector<korneeva_e_tbb::Complex>>& dense) {
  int rows = static_cast<int>(dense.size());
  int cols = dense.empty() ? 0 : static_cast<int>(dense[0].size());
  korneeva_e_tbb::SparseMatrixCCS matrix(rows, cols, 0);

  matrix.col_offsets.resize(cols + 1, 0);
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      if (dense[i][j] != korneeva_e_tbb::Complex(0.0, 0.0)) {
        matrix.values.push_back(dense[i][j]);
        matrix.row_indices.push_back(i);
        matrix.nnz++;
      }
    }
    matrix.col_offsets[j + 1] = static_cast<int>(matrix.values.size());
  }
  return matrix;
}
}  // namespace

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_incompatible_sizes) {
  korneeva_e_tbb::SparseMatrixCCS m1(2, 3, 0);
  m1.col_offsets = {0, 0, 0, 0};
  korneeva_e_tbb::SparseMatrixCCS m2(2, 2, 0);
  m2.col_offsets = {0, 0, 0};
  korneeva_e_tbb::SparseMatrixCCS result;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m1));
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m2));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));

  korneeva_e_tbb::SparseMatrixMultComplexCCS task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_negative_dimensions) {
  korneeva_e_tbb::SparseMatrixCCS m1(-1, 2, 0);
  m1.col_offsets = {0, 0, 0};
  korneeva_e_tbb::SparseMatrixCCS m2(2, 2, 0);
  m2.col_offsets = {0, 0, 0};
  korneeva_e_tbb::SparseMatrixCCS result;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m1));
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m2));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));

  korneeva_e_tbb::SparseMatrixMultComplexCCS task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_empty_input) {
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  korneeva_e_tbb::SparseMatrixCCS result;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));

  korneeva_e_tbb::SparseMatrixMultComplexCCS task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_no_output) {
  korneeva_e_tbb::SparseMatrixCCS m1(2, 2, 0);
  m1.col_offsets = {0, 0, 0};
  korneeva_e_tbb::SparseMatrixCCS m2(2, 2, 0);
  m2.col_offsets = {0, 0, 0};

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m1));
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m2));

  korneeva_e_tbb::SparseMatrixMultComplexCCS task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_identity_mult) {
  auto m1 = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                                {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(1.0, 0.0)}});
  auto m2 = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                                {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(1.0, 0.0)}});
  korneeva_e_tbb::SparseMatrixCCS result;

  RunTask(m1, m2, result);

  auto expected = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                                      {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(1.0, 0.0)}});
  ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_left_identity_mult) {
  auto i = CreateCcsFromDense(
      {{korneeva_e_tbb::Complex(1.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
       {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(1.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
       {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(1.0, 0.0)}});

  auto a = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 2.0), korneeva_e_tbb::Complex(3.0, 4.0)},
                               {korneeva_e_tbb::Complex(5.0, 0.0), korneeva_e_tbb::Complex(7.0, 8.0)},
                               {korneeva_e_tbb::Complex(9.0, 10.0), korneeva_e_tbb::Complex(11.0, 12.0)}});

  korneeva_e_tbb::SparseMatrixCCS result;

  RunTask(i, a, result);

  ExpectMatrixEq(result, a);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_zero_matrix) {
  auto m1 = CreateCcsFromDense({{korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                                {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)}});
  auto m2 = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                                {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(1.0, 0.0)}});
  korneeva_e_tbb::SparseMatrixCCS result;

  RunTask(m1, m2, result);

  auto expected = CreateCcsFromDense({{korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                                      {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)}});
  ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_full_zero_matrix) {
  auto m1 = CreateCcsFromDense({{korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                                {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)}});
  auto m2 = CreateCcsFromDense({{korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                                {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)}});
  korneeva_e_tbb::SparseMatrixCCS result;

  RunTask(m1, m2, result);

  auto expected = CreateCcsFromDense({{korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                                      {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)}});
  ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_diagonal_matrices) {
  auto m1 = CreateCcsFromDense({{korneeva_e_tbb::Complex(2.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                                {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(3.0, 0.0)}});
  auto m2 = CreateCcsFromDense({{korneeva_e_tbb::Complex(4.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                                {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(5.0, 0.0)}});
  korneeva_e_tbb::SparseMatrixCCS result;

  RunTask(m1, m2, result);

  auto expected = CreateCcsFromDense({{korneeva_e_tbb::Complex(8.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                                      {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(15.0, 0.0)}});
  ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_complex_numbers) {
  auto m1 = CreateCcsFromDense({{korneeva_e_tbb::Complex(0.0, 1.0)}});
  auto m2 = CreateCcsFromDense({{korneeva_e_tbb::Complex(0.0, -1.0)}});
  korneeva_e_tbb::SparseMatrixCCS result;

  RunTask(m1, m2, result);

  auto expected = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 0.0)}});
  ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_large_complex_values) {
  auto m1 = CreateCcsFromDense({{korneeva_e_tbb::Complex(1e10, 1e10)}});
  auto m2 = CreateCcsFromDense({{korneeva_e_tbb::Complex(1e10, -1e10)}});
  korneeva_e_tbb::SparseMatrixCCS result;

  RunTask(m1, m2, result);

  auto expected = CreateCcsFromDense({{korneeva_e_tbb::Complex(2e20, 0.0)}});
  ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_rectangular_matrices) {
  auto m1 = CreateCcsFromDense(
      {{korneeva_e_tbb::Complex(1.0, 0.0), korneeva_e_tbb::Complex(2.0, 0.0), korneeva_e_tbb::Complex(3.0, 0.0)},
       {korneeva_e_tbb::Complex(4.0, 0.0), korneeva_e_tbb::Complex(5.0, 0.0), korneeva_e_tbb::Complex(6.0, 0.0)}});
  auto m2 = CreateCcsFromDense({{korneeva_e_tbb::Complex(7.0, 0.0), korneeva_e_tbb::Complex(8.0, 0.0)},
                                {korneeva_e_tbb::Complex(9.0, 0.0), korneeva_e_tbb::Complex(10.0, 0.0)},
                                {korneeva_e_tbb::Complex(11.0, 0.0), korneeva_e_tbb::Complex(12.0, 0.0)}});
  korneeva_e_tbb::SparseMatrixCCS result;

  RunTask(m1, m2, result);

  auto expected = CreateCcsFromDense({{korneeva_e_tbb::Complex(58.0, 0.0), korneeva_e_tbb::Complex(64.0, 0.0)},
                                      {korneeva_e_tbb::Complex(139.0, 0.0), korneeva_e_tbb::Complex(154.0, 0.0)}});
  ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_matrix_vector_mult) {
  auto m1 = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 2.0), korneeva_e_tbb::Complex(3.0, 4.0)},
                                {korneeva_e_tbb::Complex(5.0, 6.0), korneeva_e_tbb::Complex(7.0, 8.0)}});
  auto vec = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 0.0)}, {korneeva_e_tbb::Complex(2.0, 0.0)}});
  korneeva_e_tbb::SparseMatrixCCS result;

  RunTask(m1, vec, result);

  auto expected = CreateCcsFromDense({{korneeva_e_tbb::Complex(7.0, 10.0)}, {korneeva_e_tbb::Complex(19.0, 22.0)}});
  ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_vector_matrix_mult) {
  auto vec = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 0.0), korneeva_e_tbb::Complex(2.0, 0.0)}});
  auto m2 = CreateCcsFromDense({{korneeva_e_tbb::Complex(3.0, 0.0), korneeva_e_tbb::Complex(4.0, 0.0)},
                                {korneeva_e_tbb::Complex(5.0, 0.0), korneeva_e_tbb::Complex(6.0, 0.0)}});
  korneeva_e_tbb::SparseMatrixCCS result;

  RunTask(vec, m2, result);

  auto expected = CreateCcsFromDense({{korneeva_e_tbb::Complex(13.0, 0.0), korneeva_e_tbb::Complex(16.0, 0.0)}});
  ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_matrix_unit_vector) {
  auto m1 = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 1.0), korneeva_e_tbb::Complex(2.0, 2.0)},
                                {korneeva_e_tbb::Complex(3.0, 3.0), korneeva_e_tbb::Complex(4.0, 4.0)}});
  auto vec = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 0.0)}, {korneeva_e_tbb::Complex(0.0, 0.0)}});
  korneeva_e_tbb::SparseMatrixCCS result;

  RunTask(m1, vec, result);

  auto expected = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 1.0)}, {korneeva_e_tbb::Complex(3.0, 3.0)}});
  ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_sparse_matrices) {
  auto m1 = CreateCcsFromDense(
      {{korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(1.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
       {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(2.0, 0.0)}});
  auto m2 = CreateCcsFromDense({{korneeva_e_tbb::Complex(3.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                                {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(4.0, 0.0)},
                                {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)}});
  korneeva_e_tbb::SparseMatrixCCS result;

  RunTask(m1, m2, result);

  auto expected = CreateCcsFromDense({{korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(4.0, 0.0)},
                                      {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)}});
  ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_mixed_values) {
  auto m1 = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                                {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(2.0, 0.0)}});
  auto m2 = CreateCcsFromDense({{korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(3.0, 0.0)},
                                {korneeva_e_tbb::Complex(4.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)}});
  korneeva_e_tbb::SparseMatrixCCS result;

  RunTask(m1, m2, result);

  auto expected = CreateCcsFromDense({{korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(3.0, 0.0)},
                                      {korneeva_e_tbb::Complex(8.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)}});
  ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_dense_sparse_mult) {
  auto m1 = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 0.0), korneeva_e_tbb::Complex(2.0, 0.0)},
                                {korneeva_e_tbb::Complex(3.0, 0.0), korneeva_e_tbb::Complex(4.0, 0.0)}});
  auto m2 = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                                {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)}});
  korneeva_e_tbb::SparseMatrixCCS result;

  RunTask(m1, m2, result);

  auto expected = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                                      {korneeva_e_tbb::Complex(3.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)}});
  ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_random_matrices1) {
  auto m1 = CreateRandomMatrix(2, 2, 2);
  auto m2 = CreateRandomMatrix(2, 2, 2);
  korneeva_e_tbb::SparseMatrixCCS result;

  RunTask(m1, m2, result);

  ASSERT_EQ(result.rows, 2);
  ASSERT_EQ(result.cols, 2);
  EXPECT_LE(result.nnz, 4);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_random_matrices2) {
  auto m1 = CreateRandomMatrix(100, 100, 500);
  auto m2 = CreateRandomMatrix(100, 100, 500);
  korneeva_e_tbb::SparseMatrixCCS result;

  RunTask(m1, m2, result);

  ASSERT_EQ(result.rows, 100);
  ASSERT_EQ(result.cols, 100);
  EXPECT_LE(result.nnz, 10000);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_tbb, test_associativity) {
  auto a = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 0.0), korneeva_e_tbb::Complex(2.0, 0.0)},
                               {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(3.0, 0.0)}});
  auto b = CreateCcsFromDense({{korneeva_e_tbb::Complex(4.0, 0.0), korneeva_e_tbb::Complex(0.0, 0.0)},
                               {korneeva_e_tbb::Complex(5.0, 0.0), korneeva_e_tbb::Complex(6.0, 0.0)}});
  auto c = CreateCcsFromDense({{korneeva_e_tbb::Complex(1.0, 0.0), korneeva_e_tbb::Complex(2.0, 0.0)},
                               {korneeva_e_tbb::Complex(0.0, 0.0), korneeva_e_tbb::Complex(1.0, 0.0)}});

  korneeva_e_tbb::SparseMatrixCCS ab;
  korneeva_e_tbb::SparseMatrixCCS ab_c;
  korneeva_e_tbb::SparseMatrixCCS bc;
  korneeva_e_tbb::SparseMatrixCCS a_bc;

  RunTask(a, b, ab);
  RunTask(ab, c, ab_c);

  RunTask(b, c, bc);
  RunTask(a, bc, a_bc);

  ExpectMatrixEq(ab_c, a_bc);
}
