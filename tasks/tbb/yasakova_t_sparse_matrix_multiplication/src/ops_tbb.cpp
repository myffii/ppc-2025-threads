#include "tbb/yasakova_t_sparse_matrix_multiplication/include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_for.h>
#include <tbb/tbb.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

void yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix::InsertElement(int row, ComplexNumber value,
                                                                                        int col) {
  bool found = false;
  for (int j = rowPointers[row]; j < rowPointers[row + 1]; j++) {
    if (columnIndices[j] == col) {
      nonZeroValues[j] += value;
      found = true;
      break;
    }
  }
  if (!found) {
    columnIndices.emplace_back(col);
    nonZeroValues.emplace_back(value);
    for (int i = row + 1; i <= rowCount; i++) {
      rowPointers[i]++;
    }
  }
}

void yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix::DisplayMatrix(
    const CompressedRowStorageMatrix& matrix) {
  for (int i = 0; i < matrix.rowCount; i++) {
    for (int j = matrix.rowPointers[i]; j < matrix.rowPointers[i + 1]; j++) {
      std::cout << "Element at (" << i << ", " << matrix.columnIndices[j] << ") = " << matrix.nonZeroValues[j] << '\n';
    }
  }
}

bool yasakova_t_sparse_matrix_multiplication::AreEqualElems(const ComplexNumber& first_matrix,
                                                            const ComplexNumber& second_matrix, double tolerance) {
  return std::abs(first_matrix.real() - second_matrix.real()) < tolerance &&
         std::abs(first_matrix.imag() - second_matrix.imag()) < tolerance;
}

std::vector<ComplexNumber> yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(
    const CompressedRowStorageMatrix& mat) {
  std::vector<ComplexNumber> actual_result = {};
  actual_result.reserve(5 + mat.nonZeroValues.size() + mat.columnIndices.size() + mat.rowPointers.size());
  actual_result.emplace_back((double)mat.rowCount);
  actual_result.emplace_back((double)mat.columnCount);
  actual_result.emplace_back((double)mat.nonZeroValues.size());
  actual_result.emplace_back((double)mat.columnIndices.size());
  actual_result.emplace_back((double)mat.rowPointers.size());
  for (unsigned int i = 0; i < (unsigned int)mat.nonZeroValues.size(); i++) {
    actual_result.emplace_back(mat.nonZeroValues[i]);
  }
  for (unsigned int i = 0; i < (unsigned int)mat.columnIndices.size(); i++) {
    actual_result.emplace_back(mat.columnIndices[i]);
  }
  for (unsigned int i = 0; i < (unsigned int)mat.rowPointers.size(); i++) {
    actual_result.emplace_back(mat.rowPointers[i]);
  }
  return actual_result;
}

bool yasakova_t_sparse_matrix_multiplication::CompareMatrices(
    const yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix& first_matrix,
    const yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix& second_matrix) {
  if (first_matrix.columnCount != second_matrix.columnCount || first_matrix.rowCount != second_matrix.rowCount) {
    return false;
  }
  for (unsigned int i = 0; i < (unsigned int)first_matrix.rowCount; i++) {
    unsigned int this_row_start = first_matrix.rowPointers[i];
    unsigned int this_row_end = first_matrix.rowPointers[i + 1];
    unsigned int other_row_start = second_matrix.rowPointers[i];
    unsigned int other_row_end = second_matrix.rowPointers[i + 1];
    if ((this_row_end - this_row_start) != (other_row_end - other_row_start)) {
      return false;
    }
    for (unsigned int j = this_row_start; j < this_row_end; j++) {
      bool found = false;
      for (unsigned int k = other_row_start; k < other_row_end; k++) {
        if (first_matrix.columnIndices[j] == second_matrix.columnIndices[k] &&
            AreEqualElems(first_matrix.nonZeroValues[j], second_matrix.nonZeroValues[k], 0.000001)) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }
  }
  return true;
}

yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix
yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(std::vector<ComplexNumber>& vec) {
  CompressedRowStorageMatrix actual_result;
  actual_result.rowCount = (int)vec[0].real();
  actual_result.columnCount = (int)vec[1].real();
  auto values_size = (unsigned int)vec[2].real();
  auto col_indices_size = (unsigned int)vec[3].real();
  auto row_ptr_size = (unsigned int)vec[4].real();
  actual_result.nonZeroValues.reserve(values_size);
  actual_result.columnIndices.reserve(col_indices_size);
  actual_result.rowPointers.reserve(row_ptr_size);
  for (unsigned int i = 0; i < values_size; i++) {
    actual_result.nonZeroValues.emplace_back(vec[5 + i]);
  }
  for (unsigned int i = 0; i < col_indices_size; i++) {
    actual_result.columnIndices.emplace_back((int)vec[5 + values_size + i].real());
  }
  for (unsigned int i = 0; i < row_ptr_size; i++) {
    actual_result.rowPointers.emplace_back((int)vec[5 + values_size + col_indices_size + i].real());
  }
  return actual_result;
}

bool yasakova_t_sparse_matrix_multiplication::TestTaskTBB::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<ComplexNumber*>(task_data->inputs[0]);
  inputData_ = std::vector<ComplexNumber>(in_ptr, in_ptr + input_size);
  std::vector<ComplexNumber> matrix_a = {};
  std::vector<ComplexNumber> matrix_b = {};
  matrix_a.reserve(5 + (unsigned int)(inputData_[2].real() + inputData_[3].real() + inputData_[4].real()));
  matrix_b.reserve(inputData_.size() -
                   (unsigned int)(5 + inputData_[2].real() + inputData_[3].real() + inputData_[4].real()));
  for (unsigned int i = 0; i < (unsigned int)(5 + inputData_[2].real() + inputData_[3].real() + inputData_[4].real());
       i++) {
    matrix_a.emplace_back(inputData_[i]);
  }
  for (auto i = (unsigned int)(5 + inputData_[2].real() + inputData_[3].real() + inputData_[4].real());
       i < (unsigned int)inputData_.size(); i++) {
    matrix_b.emplace_back(inputData_[i]);
  }
  firstMatrix_ = ConvertVectorToMatrix(matrix_a);
  secondMatrix_ = ConvertVectorToMatrix(matrix_b);
  return true;
}

bool yasakova_t_sparse_matrix_multiplication::TestTaskTBB::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<ComplexNumber*>(task_data->inputs[0]);
  std::vector<ComplexNumber> vec = std::vector<ComplexNumber>(in_ptr, in_ptr + input_size);
  return !(vec[1] != vec[5 + (int)(vec[2].real() + vec[3].real() + vec[4].real())].real());
}

bool yasakova_t_sparse_matrix_multiplication::TestTaskTBB::RunImpl() {
  CompressedRowStorageMatrix expected_result(firstMatrix_.rowCount, secondMatrix_.columnCount);

  std::vector<std::vector<std::pair<int, ComplexNumber>>> local_results(firstMatrix_.rowCount);

  tbb::parallel_for(
      tbb::blocked_range<unsigned int>(0, firstMatrix_.rowCount), [&](const tbb::blocked_range<unsigned int>& r) {
        for (unsigned int i = r.begin(); i < r.end(); ++i) {
          for (unsigned int j = firstMatrix_.rowPointers[i]; j < (unsigned int)firstMatrix_.rowPointers[i + 1]; ++j) {
            unsigned int col_a = firstMatrix_.columnIndices[j];
            ComplexNumber value_a = firstMatrix_.nonZeroValues[j];
            for (unsigned int k = secondMatrix_.rowPointers[col_a];
                 k < (unsigned int)secondMatrix_.rowPointers[col_a + 1]; ++k) {
              unsigned int col_b = secondMatrix_.columnIndices[k];
              ComplexNumber value_b = secondMatrix_.nonZeroValues[k];
              local_results[i].emplace_back(col_b, value_a * value_b);
            }
          }
        }
      });

  for (size_t row_index = 0; row_index < local_results.size(); ++row_index) {
    for (const auto& [col_index, value] : local_results[row_index]) {
      expected_result.InsertElement(static_cast<int>(row_index), value, col_index);
    }
  }

  resultData_ = ConvertMatrixToVector(expected_result);

  return true;
}

bool yasakova_t_sparse_matrix_multiplication::TestTaskTBB::PostProcessingImpl() {
  for (size_t i = 0; i < resultData_.size(); i++) {
    reinterpret_cast<ComplexNumber*>(task_data->outputs[0])[i] = resultData_[i];
  }
  return true;
}