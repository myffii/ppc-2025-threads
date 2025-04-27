#include "stl/nasedkin_e_strassen_algorithm/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <future>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace nasedkin_e_strassen_algorithm_stl {

bool StrassenStl::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* in_ptr_b = reinterpret_cast<double*>(task_data->inputs[1]);

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

  output_matrix_.resize(matrix_size_ * matrix_size_, 0.0);
  return true;
}

bool StrassenStl::ValidationImpl() {
  unsigned int input_size_a = task_data->inputs_count[0];
  unsigned int input_size_b = task_data->inputs_count[1];
  unsigned int output_size = task_data->outputs_count[0];

  if (input_size_a == 0 || input_size_b == 0 || output_size == 0) {
    return false;
  }

  int size_a = static_cast<int>(std::sqrt(input_size_a));
  int size_b = static_cast<int>(std::sqrt(input_size_b));
  int size_output = static_cast<int>(std::sqrt(output_size));

  return (size_a == size_b) && (size_a == size_output);
}

bool StrassenStl::RunImpl() {
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_);
  return true;
}

bool StrassenStl::PostProcessingImpl() {
  if (original_size_ != matrix_size_) {
    output_matrix_ = TrimMatrixToOriginalSize(output_matrix_, original_size_, matrix_size_);
  }

  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(output_matrix_, out_ptr);
  return true;
}

std::vector<double> StrassenStl::AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::plus<>());
  return result;
}

std::vector<double> StrassenStl::SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b,
                                                  int size) {
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::minus<>());
  return result;
}

std::vector<double> StandardMultiply(const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::vector<double> result(size * size, 0.0);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        result[(i * size) + j] += a[(i * size) + k] * b[(k * size) + j];
      }
    }
  }
  return result;
}

std::vector<double> StrassenStl::PadMatrixToPowerOfTwo(const std::vector<double>& matrix, int original_size) {
  int new_size = 1;
  while (new_size < original_size) {
    new_size *= 2;
  }

  std::vector<double> padded_matrix(new_size * new_size, 0);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * original_size, matrix.begin() + (i + 1) * original_size,
                      padded_matrix.begin() + i * new_size);
  }
  return padded_matrix;
}

std::vector<double> StrassenStl::TrimMatrixToOriginalSize(const std::vector<double>& matrix, int original_size,
                                                          int padded_size) {
  std::vector<double> trimmed_matrix(original_size * original_size);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * padded_size, matrix.begin() + i * padded_size + original_size,
                      trimmed_matrix.begin() + i * original_size);
  }
  return trimmed_matrix;
}

std::vector<double> StrassenStl::StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b,
                                                  int size) {
  if (size <= 32) {
    return StandardMultiply(a, b, size);
  }

  const int half_size = size / 2;
  const int tile_size = half_size * half_size;

  // Выделяем память для подматриц
  std::vector<double> a11(tile_size), a12(tile_size), a21(tile_size), a22(tile_size);
  std::vector<double> b11(tile_size), b12(tile_size), b21(tile_size), b22(tile_size);

  // Параллельное разделение матриц a и b
  auto split_a = std::async(std::launch::async, [&]() {
    SplitMatrix(a, a11, 0, 0, size);
    SplitMatrix(a, a12, 0, half_size, size);
    SplitMatrix(a, a21, half_size, 0, size);
    SplitMatrix(a, a22, half_size, half_size, size);
  });

  auto split_b = std::async(std::launch::async, [&]() {
    SplitMatrix(b, b11, 0, 0, size);
    SplitMatrix(b, b12, 0, half_size, size);
    SplitMatrix(b, b21, half_size, 0, size);
    SplitMatrix(b, b22, half_size, half_size, size);
  });

  split_a.wait();
  split_b.wait();

  // Вычисляем промежуточные матрицы параллельно
  auto p1 = std::async(std::launch::async, [&]() {
    return StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size);
  });

  auto p2 = std::async(std::launch::async,
                       [&]() { return StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size); });

  auto p3 = std::async(std::launch::async,
                       [&]() { return StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size); });

  auto p4 = std::async(std::launch::async,
                       [&]() { return StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size); });

  auto p5 = std::async(std::launch::async,
                       [&]() { return StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size); });

  auto p6 = std::async(std::launch::async, [&]() {
    return StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size);
  });

  auto p7 = std::async(std::launch::async, [&]() {
    return StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size);
  });

  // Ждем завершения всех умножений
  auto c11_task = std::async(std::launch::async, [&]() {
    return AddMatrices(SubtractMatrices(AddMatrices(p1.get(), p4.get(), half_size), p5.get(), half_size), p7.get(),
                       half_size);
  });

  auto c12_task = std::async(std::launch::async, [&]() { return AddMatrices(p3.get(), p5.get(), half_size); });

  auto c21_task = std::async(std::launch::async, [&]() { return AddMatrices(p2.get(), p4.get(), half_size); });

  auto c22_task = std::async(std::launch::async, [&]() {
    return AddMatrices(SubtractMatrices(AddMatrices(p1.get(), p3.get(), half_size), p2.get(), half_size), p6.get(),
                       half_size);
  });

  // Собираем результат
  std::vector<double> result(size * size);

  auto merge_c11 = std::async(std::launch::async, [&]() { MergeMatrix(result, c11_task.get(), 0, 0, size); });

  auto merge_c12 = std::async(std::launch::async, [&]() { MergeMatrix(result, c12_task.get(), 0, half_size, size); });

  auto merge_c21 = std::async(std::launch::async, [&]() { MergeMatrix(result, c21_task.get(), half_size, 0, size); });

  auto merge_c22 =
      std::async(std::launch::async, [&]() { MergeMatrix(result, c22_task.get(), half_size, half_size, size); });

  merge_c11.wait();
  merge_c12.wait();
  merge_c21.wait();
  merge_c22.wait();

  return result;
}

void StrassenStl::SplitMatrix(const std::vector<double>& parent, std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(parent.begin() + (row_start + i) * parent_size + col_start,
                      parent.begin() + (row_start + i) * parent_size + col_start + child_size,
                      child.begin() + i * child_size);
  }
}

void StrassenStl::MergeMatrix(std::vector<double>& parent, const std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(child.begin() + i * child_size, child.begin() + (i + 1) * child_size,
                      parent.begin() + (row_start + i) * parent_size + col_start);
  }
}

}  // namespace nasedkin_e_strassen_algorithm_stl