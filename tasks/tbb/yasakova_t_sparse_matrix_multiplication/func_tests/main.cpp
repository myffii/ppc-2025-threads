#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/yasakova_t_sparse_matrix_multiplication/include/ops_tbb.hpp"

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyRealMatrices) {
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(3, 3);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(1, 0), 0);
  first_matrix.InsertElement(0, ComplexNumber(2, 0), 2);
  first_matrix.InsertElement(1, ComplexNumber(3, 0), 1);
  first_matrix.InsertElement(2, ComplexNumber(4, 0), 0);
  first_matrix.InsertElement(2, ComplexNumber(5, 0), 1);

  second_matrix.InsertElement(0, ComplexNumber(6, 0), 1);
  second_matrix.InsertElement(1, ComplexNumber(7, 0), 0);
  second_matrix.InsertElement(2, ComplexNumber(8, 0), 2);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  expected_result.InsertElement(0, ComplexNumber(6, 0), 1);
  expected_result.InsertElement(0, ComplexNumber(16, 0), 2);
  expected_result.InsertElement(1, ComplexNumber(21, 0), 0);
  expected_result.InsertElement(2, ComplexNumber(24, 0), 1);
  expected_result.InsertElement(2, ComplexNumber(35, 0), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, InvalidMatrixDimensions) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(5, 3);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(1, 0), 0);
  first_matrix.InsertElement(0, ComplexNumber(2, 0), 2);
  first_matrix.InsertElement(1, ComplexNumber(3, 0), 1);
  first_matrix.InsertElement(2, ComplexNumber(4, 0), 0);
  first_matrix.InsertElement(2, ComplexNumber(5, 0), 1);

  second_matrix.InsertElement(0, ComplexNumber(6, 0), 1);
  second_matrix.InsertElement(1, ComplexNumber(7, 0), 0);
  second_matrix.InsertElement(2, ComplexNumber(8, 0), 2);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), false);
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyComplexMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(3, 3);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(1, 1), 0);
  first_matrix.InsertElement(0, ComplexNumber(2, 2), 2);
  first_matrix.InsertElement(1, ComplexNumber(3, 3), 1);
  first_matrix.InsertElement(2, ComplexNumber(4, 4), 0);
  first_matrix.InsertElement(2, ComplexNumber(5, 5), 1);

  second_matrix.InsertElement(0, ComplexNumber(6, 6), 1);
  second_matrix.InsertElement(1, ComplexNumber(7, 7), 0);
  second_matrix.InsertElement(2, ComplexNumber(8, 8), 2);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  expected_result.InsertElement(0, ComplexNumber(0, 12), 1);
  expected_result.InsertElement(0, ComplexNumber(0, 32), 2);
  expected_result.InsertElement(1, ComplexNumber(0, 42), 0);
  expected_result.InsertElement(2, ComplexNumber(0, 48), 1);
  expected_result.InsertElement(2, ComplexNumber(0, 70), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyRectangularMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(2, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(3, 4);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(2, 4);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(1, 0), 1);
  first_matrix.InsertElement(0, ComplexNumber(2, 0), 2);
  first_matrix.InsertElement(1, ComplexNumber(3, 0), 1);

  second_matrix.InsertElement(0, ComplexNumber(3, 0), 2);
  second_matrix.InsertElement(1, ComplexNumber(5, 0), 0);
  second_matrix.InsertElement(1, ComplexNumber(4, 0), 3);
  second_matrix.InsertElement(2, ComplexNumber(7, 0), 0);
  second_matrix.InsertElement(2, ComplexNumber(8, 0), 1);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  expected_result.InsertElement(0, ComplexNumber(19, 0), 0);
  expected_result.InsertElement(0, ComplexNumber(4, 0), 3);
  expected_result.InsertElement(0, ComplexNumber(16, 0), 1);
  expected_result.InsertElement(1, ComplexNumber(15, 0), 0);
  expected_result.InsertElement(1, ComplexNumber(12, 0), 3);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyMatricesWithNegativeElements) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(2, 2);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(2, 2);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(2, 2);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(-1, -1), 0);
  first_matrix.InsertElement(1, ComplexNumber(3, 3), 1);

  second_matrix.InsertElement(0, ComplexNumber(6, 6), 1);
  second_matrix.InsertElement(1, ComplexNumber(-7, -7), 0);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  expected_result.InsertElement(0, ComplexNumber(0, -12), 1);
  expected_result.InsertElement(1, ComplexNumber(0, -42), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyMatricesWithDoubleElements) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(2, 2);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(2, 2);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(2, 2);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(-1.7, -1.5), 0);
  first_matrix.InsertElement(1, ComplexNumber(3.7, 3.1), 1);

  second_matrix.InsertElement(0, ComplexNumber(6.3, 6.1), 1);
  second_matrix.InsertElement(1, ComplexNumber(-7.4, -7.7), 0);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  expected_result.InsertElement(0, ComplexNumber(-1.56, -19.82), 1);
  expected_result.InsertElement(1, ComplexNumber(-3.51, -51.43), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyRowByColumnMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(1, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(3, 1);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(1, 1);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(-1, 0), 0);
  first_matrix.InsertElement(0, ComplexNumber(-2, 0), 1);
  first_matrix.InsertElement(0, ComplexNumber(-3, 0), 2);

  second_matrix.InsertElement(0, ComplexNumber(1, 0), 0);
  second_matrix.InsertElement(1, ComplexNumber(2, 0), 0);
  second_matrix.InsertElement(2, ComplexNumber(3, 0), 0);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  expected_result.InsertElement(0, ComplexNumber(-14, 0), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyDiagonalMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(3, 3);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(-1, 0), 0);
  first_matrix.InsertElement(1, ComplexNumber(-2, 0), 1);
  first_matrix.InsertElement(2, ComplexNumber(-3, 0), 2);

  second_matrix.InsertElement(0, ComplexNumber(1, 0), 0);
  second_matrix.InsertElement(1, ComplexNumber(2, 0), 1);
  second_matrix.InsertElement(2, ComplexNumber(3, 0), 2);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  expected_result.InsertElement(0, ComplexNumber(-1, 0), 0);
  expected_result.InsertElement(1, ComplexNumber(-4, 0), 1);
  expected_result.InsertElement(2, ComplexNumber(-9, 0), 2);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyImaginaryMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(3, 3);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(0, 1), 0);
  first_matrix.InsertElement(0, ComplexNumber(0, 2), 2);
  first_matrix.InsertElement(1, ComplexNumber(0, 3), 1);
  first_matrix.InsertElement(2, ComplexNumber(0, 4), 0);
  first_matrix.InsertElement(2, ComplexNumber(0, 5), 1);

  second_matrix.InsertElement(0, ComplexNumber(0, 6), 1);
  second_matrix.InsertElement(1, ComplexNumber(0, 7), 0);
  second_matrix.InsertElement(2, ComplexNumber(0, 8), 2);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  expected_result.InsertElement(0, ComplexNumber(-6, 0), 1);
  expected_result.InsertElement(0, ComplexNumber(-16, 0), 2);
  expected_result.InsertElement(1, ComplexNumber(-21, 0), 0);
  expected_result.InsertElement(2, ComplexNumber(-24, 0), 1);
  expected_result.InsertElement(2, ComplexNumber(-35, 0), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyLargeSparseMatrices) {
  // Инициализация генератора случайных чисел
  std::srand(std::time(nullptr));

  // Create large sparse matrices (1000x1000 with only 1% non-zero elements)
  const int size = 1000;
  const double sparsity = 0.01;
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(size, size);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(size, size);
  std::vector<ComplexNumber> input_data;
  std::vector<ComplexNumber> output_data(size * size * 2, 0);

  // Fill first matrix with random sparse data
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      if (std::rand() / (double)RAND_MAX < sparsity) {
        double real = (std::rand() / (double)RAND_MAX * 10.0) - 5.0;
        double imag = (std::rand() / (double)RAND_MAX * 10.0) - 5.0;
        first_matrix.InsertElement(i, ComplexNumber(real, imag), j);
      }
    }
  }

  // Fill second matrix with random sparse data
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      if (std::rand() / (double)RAND_MAX < sparsity) {
        double real = (std::rand() / (double)RAND_MAX * 10.0) - 5.0;
        double imag = (std::rand() / (double)RAND_MAX * 10.0) - 5.0;
        second_matrix.InsertElement(i, ComplexNumber(real, imag), j);
      }
    }
  }

  // Convert matrices to vectors and prepare input data
  auto first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  auto second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);

  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  input_data.insert(input_data.end(), first_matrix_data.begin(), first_matrix_data.end());
  input_data.insert(input_data.end(), second_matrix_data.begin(), second_matrix_data.end());

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create and run task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);

  auto start_time = std::chrono::high_resolution_clock::now();
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "Large sparse matrices multiplication took " << duration.count() << " ms\n";

  // Basic verification - check that output is not all zeros
  bool all_zeros = true;
  for (const auto &val : output_data) {
    if (val.real() != 0.0 || val.imag() != 0.0) {
      all_zeros = false;
      break;
    }
  }
  ASSERT_FALSE(all_zeros);
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyByIdentityMatrix) {
  // Создаем произвольную разреженную матрицу 3x3
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix input_matrix(3, 3);
  input_matrix.InsertElement(0, ComplexNumber(1, 2), 1);
  input_matrix.InsertElement(1, ComplexNumber(3, 4), 0);
  input_matrix.InsertElement(1, ComplexNumber(5, 6), 2);
  input_matrix.InsertElement(2, ComplexNumber(7, 8), 1);

  // Создаем единичную матрицу 3x3
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix identity_matrix(3, 3);
  identity_matrix.InsertElement(0, ComplexNumber(1, 0), 0);
  identity_matrix.InsertElement(1, ComplexNumber(1, 0), 1);
  identity_matrix.InsertElement(2, ComplexNumber(1, 0), 2);

  // Подготавливаем входные данные
  std::vector<ComplexNumber> input_data;
  auto input_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(input_matrix);
  auto identity_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(identity_matrix);

  input_data.reserve(input_matrix_data.size() + identity_matrix_data.size());
  input_data.insert(input_data.end(), input_matrix_data.begin(), input_matrix_data.end());
  input_data.insert(input_data.end(), identity_matrix_data.begin(), identity_matrix_data.end());

  // Выходной буфер
  std::vector<ComplexNumber> output_data(input_matrix.rowCount * identity_matrix.columnCount * 2, 0);

  // Создаем task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Создаем и выполняем задачу
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  // Проверяем, что результат равен исходной матрице
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix result_matrix =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);

  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(result_matrix, input_matrix));
}

TEST(yasakova_t_sparse_matrix_multiplication, ZeroMatrixTest) {
  // Создаем матрицы
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(3, 3);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  // Заполняем first_matrix нулями (sparse, поэтому явно не вставляем элементы)
  // Заполняем second_matrix нулями
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }

  // expected_result также должна быть нулевой матрицей (sparse)

  // Создаем task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Создаем Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}
