#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <functional>  // Добавлено для std::plus и std::minus
#include <vector>

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr_a = reinterpret_cast<int*>(task_data->inputs[0]);
  auto* in_ptr_b = reinterpret_cast<int*>(task_data->inputs[1]);

  matrix_size_ = static_cast<int>(std::sqrt(input_size));
  input_matrix_a_.resize(matrix_size_ * matrix_size_);
  input_matrix_b_.resize(matrix_size_ * matrix_size_);

  std::ranges::copy(in_ptr_a, in_ptr_a + input_size, input_matrix_a_.begin());
  std::ranges::copy(in_ptr_b, in_ptr_b + input_size, input_matrix_b_.begin());

  if ((matrix_size_ & (matrix_size_ - 1)) != 0) {
    original_size_ = matrix_size_;
    input_matrix_a_ = PadMatrixToPowerOfTwo(input_matrix_a_, matrix_size_);
    input_matrix_b_ = PadMatrixToPowerOfTwo(input_matrix_b_, matrix_size_);
    matrix_size_ = static_cast<int>(std::sqrt(input_matrix_a_.size()));
  } else {
    original_size_ = matrix_size_;
  }

  output_matrix_.resize(matrix_size_ * matrix_size_, 0);
  return true;
}

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::ValidationImpl() {
  unsigned int input_size_a = task_data->inputs_count[0];
  unsigned int input_size_b = task_data->inputs_count[1];
  unsigned int output_size = task_data->outputs_count[0];

  int size_a = static_cast<int>(std::sqrt(input_size_a));
  int size_b = static_cast<int>(std::sqrt(input_size_b));
  int size_output = static_cast<int>(std::sqrt(output_size));

  return (size_a == size_b) && (size_a == size_output);
}

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::RunImpl() {
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_);
  return true;
}

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::PostProcessingImpl() {
  if (original_size_ != matrix_size_) {
    output_matrix_ = TrimMatrixToOriginalSize(output_matrix_, original_size_, matrix_size_);
  }

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(output_matrix_, out_ptr);
  return true;
}

std::vector<int> nasedkin_e_strassen_algorithm_seq::StrassenSequential::AddMatrices(const std::vector<int>& a,
                                                                                    const std::vector<int>& b,
                                                                                    int size) {
  std::vector<int> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::plus<>());
  return result;
}

std::vector<int> nasedkin_e_strassen_algorithm_seq::StrassenSequential::SubtractMatrices(const std::vector<int>& a,
                                                                                         const std::vector<int>& b,
                                                                                         int size) {
  std::vector<int> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::minus<>());
  return result;
}

std::vector<int> nasedkin_e_strassen_algorithm_seq::StrassenSequential::PadMatrixToPowerOfTwo(
    const std::vector<int>& matrix, int original_size) {
  int new_size = 1;
  while (new_size < original_size) {
    new_size *= 2;
  }

  std::vector<int> padded_matrix(new_size * new_size, 0);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * original_size, matrix.begin() + (i + 1) * original_size,
                      padded_matrix.begin() + i * new_size);
  }
  return padded_matrix;
}

std::vector<int> nasedkin_e_strassen_algorithm_seq::StrassenSequential::TrimMatrixToOriginalSize(
    const std::vector<int>& matrix, int original_size, int padded_size) {
  std::vector<int> trimmed_matrix(original_size * original_size);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * padded_size, matrix.begin() + i * padded_size + original_size,
                      trimmed_matrix.begin() + i * original_size);
  }
  return trimmed_matrix;
}

std::vector<int> nasedkin_e_strassen_algorithm_seq::StrassenSequential::StandardMultiply(const std::vector<int>& a,
                                                                                         const std::vector<int>& b,
                                                                                         int size) {
  std::vector<int> result(size * size, 0);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        result[(i * size) + j] += a[(i * size) + k] * b[(k * size) + j];
      }
    }
  }
  return result;
}

std::vector<int> nasedkin_e_strassen_algorithm_seq::StrassenSequential::StrassenMultiply(const std::vector<int>& a,
                                                                                         const std::vector<int>& b,
                                                                                         int size) {
  if (size <= 32) {
    return StandardMultiply(a, b, size);
  }

  int half_size = size / 2;
  std::vector<int> a11(half_size * half_size);
  std::vector<int> a12(half_size * half_size);
  std::vector<int> a21(half_size * half_size);
  std::vector<int> a22(half_size * half_size);

  std::vector<int> b11(half_size * half_size);
  std::vector<int> b12(half_size * half_size);
  std::vector<int> b21(half_size * half_size);
  std::vector<int> b22(half_size * half_size);

  SplitMatrix(a, a11, 0, 0, size);
  SplitMatrix(a, a12, 0, half_size, size);
  SplitMatrix(a, a21, half_size, 0, size);
  SplitMatrix(a, a22, half_size, half_size, size);

  SplitMatrix(b, b11, 0, 0, size);
  SplitMatrix(b, b12, 0, half_size, size);
  SplitMatrix(b, b21, half_size, 0, size);
  SplitMatrix(b, b22, half_size, half_size, size);

  std::vector<int> p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size);
  std::vector<int> p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size);
  std::vector<int> p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size);
  std::vector<int> p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size);
  std::vector<int> p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size);
  std::vector<int> p6 =
      StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size);
  std::vector<int> p7 =
      StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size);

  std::vector<int> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  std::vector<int> c12 = AddMatrices(p3, p5, half_size);
  std::vector<int> c21 = AddMatrices(p2, p4, half_size);
  std::vector<int> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

  std::vector<int> result(size * size);
  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);

  return result;
}

void nasedkin_e_strassen_algorithm_seq::StrassenSequential::SplitMatrix(const std::vector<int>& parent,
                                                                        std::vector<int>& child, int row_start,
                                                                        int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(parent.begin() + (row_start + i) * parent_size + col_start,
                      parent.begin() + (row_start + i) * parent_size + col_start + child_size,
                      child.begin() + i * child_size);
  }
}

void nasedkin_e_strassen_algorithm_seq::StrassenSequential::MergeMatrix(std::vector<int>& parent,
                                                                        const std::vector<int>& child, int row_start,
                                                                        int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(child.begin() + i * child_size, child.begin() + (i + 1) * child_size,
                      parent.begin() + (row_start + i) * parent_size + col_start);
  }
}