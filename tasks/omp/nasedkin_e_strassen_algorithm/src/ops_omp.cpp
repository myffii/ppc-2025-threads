#include "omp/nasedkin_e_strassen_algorithm/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

namespace nasedkin_e_strassen_algorithm_omp {

bool StrassenOmp::PreProcessingImpl() {
  auto *in_ptr_a = reinterpret_cast<double *>(task_data->inputs[0]);
  auto *in_ptr_b = reinterpret_cast<double *>(task_data->inputs[1]);

  matrix_size_a_ = static_cast<int>(std::sqrt(task_data->inputs_count[0]));
  matrix_size_b_ = static_cast<int>(std::sqrt(task_data->inputs_count[1]));

  input_matrix_a_.resize(matrix_size_a_ * matrix_size_a_);
  input_matrix_b_.resize(matrix_size_b_ * matrix_size_b_);

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(task_data->inputs_count[0]); i++) {
    input_matrix_a_[i] = in_ptr_a[i];
  }
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(task_data->inputs_count[1]); i++) {
    input_matrix_b_[i] = in_ptr_b[i];
  }

  max_size_ = std::max(matrix_size_a_, matrix_size_b_);
  int padded_size = 1;
  while (padded_size < max_size_) {
    padded_size *= 2;
  }

  input_matrix_a_ = PadMatrix(input_matrix_a_, matrix_size_a_, padded_size);
  input_matrix_b_ = PadMatrix(input_matrix_b_, matrix_size_b_, padded_size);

  matrix_size_a_ = matrix_size_b_ = padded_size;
  output_matrix_.resize(padded_size * padded_size, 0.0);
  return true;
}

bool StrassenOmp::ValidationImpl() {
  return task_data->inputs_count.size() == 2 &&
         task_data->outputs_count.size() == 1;
}

bool StrassenOmp::RunImpl() {
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_,
                                    matrix_size_a_, matrix_size_b_);
  return true;
}

bool StrassenOmp::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);
#pragma omp parallel for
  for (int i = 0; i < max_size_ * max_size_; i++) {
    out_ptr[i] = output_matrix_[i];
  }
  return true;
}

std::vector<double> StrassenOmp::AddMatrices(const std::vector<double> &a,
                                             const std::vector<double> &b,
                                             int size) {
  std::vector<double> result(size * size);
#pragma omp parallel for
  for (int i = 0; i < size * size; i++) {
    result[i] = a[i] + b[i];
  }
  return result;
}

std::vector<double> StrassenOmp::SubtractMatrices(const std::vector<double> &a,
                                                  const std::vector<double> &b,
                                                  int size) {
  std::vector<double> result(size * size);
#pragma omp parallel for
  for (int i = 0; i < size * size; i++) {
    result[i] = a[i] - b[i];
  }
  return result;
}

std::vector<double> StandardMultiply(const std::vector<double> &a,
                                     const std::vector<double> &b, int size_a,
                                     int size_b) {
  int result_size = std::max(size_a, size_b);
  std::vector<double> result(result_size * result_size, 0.0);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < size_a; i++) {
    for (int j = 0; j < size_b; j++) {
      double sum = 0.0;
      for (int k = 0; k < size_a; k++) {
        sum += a[i * size_a + k] * b[k * size_b + j];
      }
      result[i * result_size + j] = sum;
    }
  }
  return result;
}

std::vector<double> StrassenOmp::StrassenMultiply(const std::vector<double> &a,
                                                  const std::vector<double> &b,
                                                  int size_a, int size_b) {
  int size = std::max(size_a, size_b);
  if (size <= 32) {
    return StandardMultiply(a, b, size_a, size_b);
  }

  int half_size = size / 2;
  std::vector<double> a11(half_size * half_size);
  std::vector<double> a12(half_size * half_size);
  std::vector<double> a21(half_size * half_size);
  std::vector<double> a22(half_size * half_size);

  std::vector<double> b11(half_size * half_size);
  std::vector<double> b12(half_size * half_size);
  std::vector<double> b21(half_size * half_size);
  std::vector<double> b22(half_size * half_size);

#pragma omp parallel sections
  {
#pragma omp section
    SplitMatrix(a, a11, 0, 0, size);
#pragma omp section
    SplitMatrix(a, a12, 0, half_size, size);
#pragma omp section
    SplitMatrix(a, a21, half_size, 0, size);
#pragma omp section
    SplitMatrix(a, a22, half_size, half_size, size);
#pragma omp section
    SplitMatrix(b, b11, 0, 0, size);
#pragma omp section
    SplitMatrix(b, b12, 0, half_size, size);
#pragma omp section
    SplitMatrix(b, b21, half_size, 0, size);
#pragma omp section
    SplitMatrix(b, b22, half_size, half_size, size);
  }

  std::vector<double> p1;
  std::vector<double> p2;
  std::vector<double> p3;
  std::vector<double> p4;
  std::vector<double> p5;
  std::vector<double> p6;
  std::vector<double> p7;

#pragma omp parallel sections
  {
#pragma omp section
    p1 = StrassenMultiply(AddMatrices(a11, a22, half_size),
                          AddMatrices(b11, b22, half_size), half_size,
                          half_size);
#pragma omp section
    p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size,
                          half_size);
#pragma omp section
    p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size,
                          half_size);
#pragma omp section
    p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size,
                          half_size);
#pragma omp section
    p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size,
                          half_size);
#pragma omp section
    p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size),
                          AddMatrices(b11, b12, half_size), half_size,
                          half_size);
#pragma omp section
    p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size),
                          AddMatrices(b21, b22, half_size), half_size,
                          half_size);
  }

  std::vector<double> c11 = AddMatrices(
      SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7,
      half_size);
  std::vector<double> c12 = AddMatrices(p3, p5, half_size);
  std::vector<double> c21 = AddMatrices(p2, p4, half_size);
  std::vector<double> c22 = AddMatrices(
      SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6,
      half_size);

  std::vector<double> result(size * size);
#pragma omp parallel sections
  {
#pragma omp section
    MergeMatrix(result, c11, 0, 0, size);
#pragma omp section
    MergeMatrix(result, c12, 0, half_size, size);
#pragma omp section
    MergeMatrix(result, c21, half_size, 0, size);
#pragma omp section
    MergeMatrix(result, c22, half_size, half_size, size);
  }

  return result;
}

void StrassenOmp::SplitMatrix(const std::vector<double> &parent,
                              std::vector<double> &child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(
        parent.begin() + (row_start + i) * parent_size + col_start,
        parent.begin() + (row_start + i) * parent_size + col_start + child_size,
        child.begin() + i * child_size);
  }
}

void StrassenOmp::MergeMatrix(std::vector<double> &parent,
                              const std::vector<double> &child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(
        child.begin() + i * child_size, child.begin() + (i + 1) * child_size,
        parent.begin() + (row_start + i) * parent_size + col_start);
  }
}

std::vector<double> StrassenOmp::PadMatrix(const std::vector<double> &matrix,
                                           int original_size, int target_size) {
  std::vector<double> padded_matrix(target_size * target_size, 0);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * original_size,
                      matrix.begin() + i * original_size +
                          std::min(original_size, target_size),
                      padded_matrix.begin() + i * target_size);
  }
  return padded_matrix;
}

std::vector<double> StrassenOmp::TrimMatrix(const std::vector<double> &matrix,
                                            int target_size) {
  std::vector<double> trimmed_matrix(target_size * target_size);
  for (int i = 0; i < target_size; ++i) {
    std::ranges::copy(matrix.begin() + i * target_size,
                      matrix.begin() + i * target_size + target_size,
                      trimmed_matrix.begin() + i * target_size);
  }
  return trimmed_matrix;
}

} // namespace nasedkin_e_strassen_algorithm_omp