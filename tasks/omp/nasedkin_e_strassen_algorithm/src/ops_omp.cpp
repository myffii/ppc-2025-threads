#include "omp/nasedkin_e_strassen_algorithm/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace nasedkin_e_strassen_algorithm_omp {

bool StrassenOmp::PreProcessingImpl() {
  auto* in_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* in_ptr_b = reinterpret_cast<double*>(task_data->inputs[1]);
  
  rows_a_ = task_data->inputs_count[0] / task_data->inputs_count[1];
  cols_a_ = task_data->inputs_count[1] / task_data->inputs_count[0];
  cols_b_ = task_data->inputs_count[1];
  
  input_matrix_a_.resize(rows_a_ * cols_a_);
  input_matrix_b_.resize(cols_a_ * cols_b_);
  
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(rows_a_ * cols_a_); i++) {
    input_matrix_a_[i] = in_ptr_a[i];
  }
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(cols_a_ * cols_b_); i++) {
    input_matrix_b_[i] = in_ptr_b[i];
  }
  
  orig_rows_a_ = rows_a_;
  orig_cols_a_ = cols_a_;
  orig_cols_b_ = cols_b_;
  
  // Находим ближайшую степень двойки для всех измерений
  int max_dim = std::max({rows_a_, cols_a_, cols_b_});
  int new_size = 1;
  while (new_size < max_dim) new_size *= 2;
  
  if (rows_a_ != new_size || cols_a_ != new_size || cols_b_ != new_size) {
    input_matrix_a_ = PadMatrix(input_matrix_a_, rows_a_, cols_a_, new_size, new_size);
    input_matrix_b_ = PadMatrix(input_matrix_b_, cols_a_, cols_b_, new_size, new_size);
    rows_a_ = cols_a_ = cols_b_ = new_size;
  }
  
  output_matrix_.resize(rows_a_ * cols_b_, 0.0);
  return true;
}

bool StrassenOmp::ValidationImpl() {
  return task_data->inputs_count[0] % task_data->inputs_count[1] == 0 &&
         task_data->outputs_count[0] == (task_data->inputs_count[0] / task_data->inputs_count[1]) * task_data->inputs_count[1];
}

bool StrassenOmp::RunImpl() {
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, rows_a_, cols_a_, cols_b_);
  return true;
}

bool StrassenOmp::PostProcessingImpl() {
  if (orig_rows_a_ != rows_a_ || orig_cols_b_ != cols_b_) {
    output_matrix_ = TrimMatrix(output_matrix_, rows_a_, cols_b_, orig_rows_a_, orig_cols_b_);
  }
  
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(output_matrix_.size()); i++) {
    out_ptr[i] = output_matrix_[i];
  }
  return true;
}

std::vector<double> StrassenOmp::AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int rows, int cols) {
  std::vector<double> result(rows * cols);
#pragma omp parallel for
  for (int i = 0; i < rows * cols; i++) {
    result[i] = a[i] + b[i];
  }
  return result;
}

std::vector<double> StrassenOmp::SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b, int rows, int cols) {
  std::vector<double> result(rows * cols);
#pragma omp parallel for
  for (int i = 0; i < rows * cols; i++) {
    result[i] = a[i] - b[i];
  }
  return result;
}

std::vector<double> StandardMultiply(const std::vector<double>& a, const std::vector<double>& b, 
                                    int rows_a, int cols_a, int cols_b) {
  std::vector<double> result(rows_a * cols_b, 0.0);
#pragma omp parallel for
  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      double sum = 0.0;
      for (int k = 0; k < cols_a; k++) {
        sum += a[i * cols_a + k] * b[k * cols_b + j];
      }
      result[i * cols_b + j] = sum;
    }
  }
  return result;
}

std::vector<double> StrassenOmp::StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b,
                                                 int rows_a, int cols_a, int cols_b) {
  if (rows_a <= 32 || cols_a <= 32 || cols_b <= 32) {
    return StandardMultiply(a, b, rows_a, cols_a, cols_b);
  }
  
  int half_rows = rows_a / 2;
  int half_cols_a = cols_a / 2;
  int half_cols_b = cols_b / 2;
  
  // Разделение матриц
  std::vector<double> a11(half_rows * half_cols_a);
  std::vector<double> a12(half_rows * half_cols_a);
  std::vector<double> a21(half_rows * half_cols_a);
  std::vector<double> a22(half_rows * half_cols_a);
  
  std::vector<double> b11(half_cols_a * half_cols_b);
  std::vector<double> b12(half_cols_a * half_cols_b);
  std::vector<double> b21(half_cols_a * half_cols_b);
  std::vector<double> b22(half_cols_a * half_cols_b);
  
#pragma omp parallel sections
  {
#pragma omp section
    SplitMatrix(a, a11, 0, 0, rows_a, cols_a);
#pragma omp section
    SplitMatrix(a, a12, 0, half_cols_a, rows_a, cols_a);
#pragma omp section
    SplitMatrix(a, a21, half_rows, 0, rows_a, cols_a);
#pragma omp section
    SplitMatrix(a, a22, half_rows, half_cols_a, rows_a, cols_a);
#pragma omp section
    SplitMatrix(b, b11, 0, 0, cols_a, cols_b);
#pragma omp section
    SplitMatrix(b, b12, 0, half_cols_b, cols_a, cols_b);
#pragma omp section
    SplitMatrix(b, b21, half_cols_a, 0, cols_a, cols_b);
#pragma omp section
    SplitMatrix(b, b22, half_cols_a, half_cols_b, cols_a, cols_b);
  }
  
  // Рекурсивные вычисления
  std::vector<double> p1, p2, p3, p4, p5, p6, p7;
#pragma omp parallel sections
  {
#pragma omp section
    p1 = StrassenMultiply(AddMatrices(a11, a22, half_rows, half_cols_a),
                         AddMatrices(b11, b22, half_cols_a, half_cols_b),
                         half_rows, half_cols_a, half_cols_b);
#pragma omp section
    p2 = StrassenMultiply(AddMatrices(a21, a22, half_rows, half_cols_a),
                         b11, half_rows, half_cols_a, half_cols_b);
#pragma omp section
    p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_cols_a, half_cols_b),
                         half_rows, half_cols_a, half_cols_b);
#pragma omp section
    p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_cols_a, half_cols_b),
                         half_rows, half_cols_a, half_cols_b);
#pragma omp section
    p5 = StrassenMultiply(AddMatrices(a11, a12, half_rows, half_cols_a),
                         b22, half_rows, half_cols_a, half_cols_b);
#pragma omp section
    p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_rows, half_cols_a),
                         AddMatrices(b11, b12, half_cols_a, half_cols_b),
                         half_rows, half_cols_a, half_cols_b);
#pragma omp section
    p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_rows, half_cols_a),
                         AddMatrices(b21, b22, half_cols_a, half_cols_b),
                         half_rows, half_cols_a, half_cols_b);
  }
  
  // Сборка результата
  std::vector<double> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_rows, half_cols_b),
                                                       p5, half_rows, half_cols_b),
                                      p7, half_rows, half_cols_b);
  std::vector<double> c12 = AddMatrices(p3, p5, half_rows, half_cols_b);
  std::vector<double> c21 = AddMatrices(p2, p4, half_rows, half_cols_b);
  std::vector<double> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_rows, half_cols_b),
                                                       p2, half_rows, half_cols_b),
                                      p6, half_rows, half_cols_b);
  
  std::vector<double> result(rows_a * cols_b);
#pragma omp parallel sections
  {
#pragma omp section
    MergeMatrix(result, c11, 0, 0, rows_a, cols_b);
#pragma omp section
    MergeMatrix(result, c12, 0, half_cols_b, rows_a, cols_b);
#pragma omp section
    MergeMatrix(result, c21, half_rows, 0, rows_a, cols_b);
#pragma omp section
    MergeMatrix(result, c22, half_rows, half_cols_b, rows_a, cols_b);
  }
  
  return result;
}

void StrassenOmp::SplitMatrix(const std::vector<double>& parent, std::vector<double>& child, 
                             int row_start, int col_start, int parent_rows, int parent_cols) {
  int child_rows = static_cast<int>(child.size()) / (parent_cols - col_start);
  int child_cols = parent_cols - col_start;
  for (int i = 0; i < child_rows; ++i) {
    std::ranges::copy(parent.begin() + (row_start + i) * parent_cols + col_start,
                     parent.begin() + (row_start + i) * parent_cols + col_start + child_cols,
                     child.begin() + i * child_cols);
  }
}

void StrassenOmp::MergeMatrix(std::vector<double>& parent, const std::vector<double>& child,
                             int row_start, int col_start, int parent_rows, int parent_cols) {
  int child_rows = static_cast<int>(child.size()) / (parent_cols - col_start);
  int child_cols = parent_cols - col_start;
  for (int i = 0; i < child_rows; ++i) {
    std::ranges::copy(child.begin() + i * child_cols, 
                     child.begin() + (i + 1) * child_cols,
                     parent.begin() + (row_start + i) * parent_cols + col_start);
  }
}

std::vector<double> StrassenOmp::PadMatrix(const std::vector<double>& matrix, int orig_rows, int orig_cols,
                                          int new_rows, int new_cols) {
  std::vector<double> padded_matrix(new_rows * new_cols, 0);
  for (int i = 0; i < orig_rows; ++i) {
    std::ranges::copy(matrix.begin() + i * orig_cols, 
                     matrix.begin() + (i + 1) * orig_cols,
                     padded_matrix.begin() + i * new_cols);
  }
  return padded_matrix;
}

std::vector<double> StrassenOmp::TrimMatrix(const std::vector<double>& matrix, int padded_rows, int padded_cols,
                                           int orig_rows, int orig_cols) {
  std::vector<double> trimmed_matrix(orig_rows * orig_cols);
  for (int i = 0; i < orig_rows; ++i) {
    std::ranges::copy(matrix.begin() + i * padded_cols,
                     matrix.begin() + i * padded_cols + orig_cols,
                     trimmed_matrix.begin() + i * orig_cols);
  }
  return trimmed_matrix;
}

}  // namespace nasedkin_e_strassen_algorithm_omp