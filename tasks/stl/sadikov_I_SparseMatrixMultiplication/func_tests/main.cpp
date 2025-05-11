#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/sadikov_I_SparseMatrixMultiplication/include/SparseMatrix.hpp"
#include "stl/sadikov_I_SparseMatrixMultiplication/include/ops_stl.hpp"

namespace {
std::vector<double> GetRandomMatrix(int size) {
  std::vector<double> data(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  int low = -5000;
  int high = 5000;
  std::uniform_int_distribution<> number(low, high);
  for (int i = 0; i < size / 5; ++i) {
    data[i] = static_cast<double>(number(gen));
  }
  std::ranges::shuffle(data, gen);
  return data;
}

struct TestData {
  std::vector<double> first_matrix;
  int first_matrix_rows_count = 0;
  int first_matrix_columns_count = 0;
  std::vector<double> second_matrix;
  int second_matrix_rows_count = 0;
  int second_matrix_columns_count = 0;
  std::vector<double> multiplication_result;
};

// NOLINTNEXTLINE(readability-identifier-naming)
class sadikov_i_matrix_multiplication_testing_stl : public testing::Test {
  static constexpr auto kEpsilon = 0.000001;

 protected:
  std::shared_ptr<ppc::core::TaskData> m_task_data_stl;

 public:
  // NOLINTNEXTLINE(readability-identifier-naming)
  sadikov_i_matrix_multiplication_testing_stl() { m_task_data_stl = std::make_shared<ppc::core::TaskData>(); }

  void FillTaskData(TestData& data) {
    m_task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.first_matrix.data()));
    m_task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.second_matrix.data()));
    m_task_data_stl->inputs_count.emplace_back(data.first_matrix_rows_count);
    m_task_data_stl->inputs_count.emplace_back(data.first_matrix_columns_count);
    m_task_data_stl->inputs_count.emplace_back(data.second_matrix_rows_count);
    m_task_data_stl->inputs_count.emplace_back(data.second_matrix_columns_count);
    m_task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(data.multiplication_result.data()));
    m_task_data_stl->outputs_count.emplace_back(data.first_matrix_rows_count * data.second_matrix_columns_count);
  }

  void RunTask(const std::vector<double>& result_checker, const TestData& data, bool alleged_validation_status) {
    sadikov_i_sparse_matrix_multiplication_task_stl::CCSMatrixSTL test_task_stl(m_task_data_stl);
    ASSERT_EQ(test_task_stl.Validation(), alleged_validation_status);
    if (alleged_validation_status) {
      test_task_stl.PreProcessing();
      test_task_stl.Run();
      test_task_stl.PostProcessing();
      for (size_t i = 0; i < result_checker.size(); ++i) {
        EXPECT_NEAR(data.multiplication_result[i], result_checker[i], kEpsilon);
      }
    }
  }
};
}  // namespace

TEST_F(sadikov_i_matrix_multiplication_testing_stl, test_rect_matrix) {
  TestData test_data = {.first_matrix = {0, 0, 0, 5.0, 2.0, 0, 1.0, 0, 7.0, 7.0, 0, 0},
                        .first_matrix_rows_count = 3,
                        .first_matrix_columns_count = 4,
                        .second_matrix = {1.0, 0, 0, 2.0, 0, 8.0, 0, 0, 0, 0, 5.0, 0},
                        .second_matrix_rows_count = 4,
                        .second_matrix_columns_count = 3,
                        .multiplication_result = std::vector<double>(12)};
  std::vector<double> test_out{0.0, 25.0, 0.0, 2.0, 0.0, 0.0, 21.0, 0.0, 56.0};
  FillTaskData(test_data);
  RunTask(test_out, test_data, true);
}

TEST_F(sadikov_i_matrix_multiplication_testing_stl, test_square_matrix) {
  TestData test_data = {.first_matrix = {1.0, 0.0, 0.0, 0.0, 7.0, 0.0, 4.0, 9.0, 0.0},
                        .first_matrix_rows_count = 3,
                        .first_matrix_columns_count = 3,
                        .second_matrix = {0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 10.0, 0.0, 0.0},
                        .second_matrix_rows_count = 3,
                        .second_matrix_columns_count = 3,
                        .multiplication_result = std::vector<double>(9)};
  std::vector<double> test_out{0.0, 0.0, 3.0, 14.0, 0.0, 0.0, 18.0, 0.0, 12.0};
  FillTaskData(test_data);
  RunTask(test_out, test_data, true);
}

TEST_F(sadikov_i_matrix_multiplication_testing_stl, test_empty_matrix) {
  TestData test_data;
  std::vector<double> test_out;
  FillTaskData(test_data);
  RunTask(test_out, test_data, true);
}

TEST_F(sadikov_i_matrix_multiplication_testing_stl, test_random_matrix) {
  constexpr int kSize = 40;
  TestData test_data = {.first_matrix = GetRandomMatrix(kSize * kSize),
                        .first_matrix_rows_count = kSize,
                        .first_matrix_columns_count = kSize,
                        .second_matrix = GetRandomMatrix(kSize * kSize),
                        .second_matrix_rows_count = kSize,
                        .second_matrix_columns_count = kSize,
                        .multiplication_result = std::vector<double>(kSize * kSize)};
  std::vector<double> test_out = sadikov_i_sparse_matrix_multiplication_task_stl::BaseMatrixMultiplication(
      test_data.first_matrix, kSize, kSize, test_data.second_matrix, kSize, kSize);
  FillTaskData(test_data);
  RunTask(test_out, test_data, true);
}

TEST_F(sadikov_i_matrix_multiplication_testing_stl, test_random_matrix2) {
  constexpr int kSize = 52;
  TestData test_data = {.first_matrix = GetRandomMatrix(kSize * kSize),
                        .first_matrix_rows_count = kSize,
                        .first_matrix_columns_count = kSize,
                        .second_matrix = GetRandomMatrix(kSize * kSize),
                        .second_matrix_rows_count = kSize,
                        .second_matrix_columns_count = kSize,
                        .multiplication_result = std::vector<double>(kSize * kSize)};
  std::vector<double> test_out = sadikov_i_sparse_matrix_multiplication_task_stl::BaseMatrixMultiplication(
      test_data.first_matrix, kSize, kSize, test_data.second_matrix, kSize, kSize);
  FillTaskData(test_data);
  RunTask(test_out, test_data, true);
}

TEST_F(sadikov_i_matrix_multiplication_testing_stl, test_wrong_matrix_size) {
  constexpr int kFirstMatrixSize = 3;
  constexpr int kSecondMatrixSize = 2;
  TestData test_data = {.first_matrix = GetRandomMatrix(kFirstMatrixSize * kFirstMatrixSize),
                        .first_matrix_rows_count = kFirstMatrixSize,
                        .first_matrix_columns_count = kFirstMatrixSize,
                        .second_matrix = GetRandomMatrix(kSecondMatrixSize * kSecondMatrixSize),
                        .second_matrix_rows_count = kSecondMatrixSize,
                        .second_matrix_columns_count = kSecondMatrixSize,
                        .multiplication_result = std::vector<double>(kFirstMatrixSize * kFirstMatrixSize)};
  std::vector<double> test_out = sadikov_i_sparse_matrix_multiplication_task_stl::BaseMatrixMultiplication(
      test_data.first_matrix, kFirstMatrixSize, kFirstMatrixSize, test_data.second_matrix, kSecondMatrixSize,
      kSecondMatrixSize);
  FillTaskData(test_data);
  RunTask(test_out, test_data, false);
}