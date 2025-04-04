#include "tbb/nasedkin_e_strassen_algorithm/include/ops_tbb.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/parallel_invoke.h"
#include "oneapi/tbb/task_group.h"

namespace nasedkin_e_strassen_algorithm_tbb {

bool StrassenTbb::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* in_ptr_b = reinterpret_cast<double*>(task_data->inputs[1]);

  matrix_size_ = static_cast<int>(std::sqrt(input_size));
  input_matrix_a_.resize(matrix_size_ * matrix_size_);
  input_matrix_b_.resize(matrix_size_ * matrix_size_);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, input_size), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i != r.end(); ++i) {
      input_matrix_a_[i] = in_ptr_a[i];
      input_matrix_b_[i] = in_ptr_b[i];
    }
  });

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

bool StrassenTbb::ValidationImpl() {
  unsigned int input_size_a = task_data->inputs_count[0];
  unsigned int input_size_b = task_data->inputs_count[1];
  unsigned int output_size = task_data->outputs_count[0];

  int size_a = static_cast<int>(std::sqrt(input_size_a));
  int size_b = static_cast<int>(std::sqrt(input_size_b));
  int size_output = static_cast<int>(std::sqrt(output_size));

  return (task_data->inputs.size() >= 2) && (size_a == size_b) && (size_a == size_output);
}

bool StrassenTbb::RunImpl() {
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_);
  return true;
}

bool StrassenTbb::PostProcessingImpl() {
  if (original_size_ != matrix_size_) {
    output_matrix_ = TrimMatrixToOriginalSize(output_matrix_, original_size_, matrix_size_);
  }

  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  // Исправлено сужающее преобразование: используем size_t вместо int
  tbb::parallel_for(tbb::blocked_range<size_t>(0, output_matrix_.size()), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i != r.end(); ++i) {
      out_ptr[i] = output_matrix_[i];
    }
  });
  return true;
}

std::vector<double> StrassenTbb::AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::vector<double> result(size * size);
  tbb::parallel_for(tbb::blocked_range<int>(0, size * size), [&](const tbb::blocked_range<int>& r) {
    for (int i = r.begin(); i != r.end(); ++i) {
      result[i] = a[i] + b[i];
    }
  });
  return result;
}

std::vector<double> StrassenTbb::SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b,
                                                  int size) {
  std::vector<double> result(size * size);
  tbb::parallel_for(tbb::blocked_range<int>(0, size * size), [&](const tbb::blocked_range<int>& r) {
    for (int i = r.begin(); i != r.end(); ++i) {
      result[i] = a[i] - b[i];
    }
  });
  return result;
}

std::vector<double> StandardMultiply(const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::vector<double> result(size * size, 0.0);
  tbb::parallel_for(tbb::blocked_range<int>(0, size), [&](const tbb::blocked_range<int>& r) {
    for (int i = r.begin(); i != r.end(); ++i) {
      for (int j = 0; j < size; j++) {
        double sum = 0.0;
        for (int k = 0; k < size; k++) {
          sum += a[(i * size) + k] * b[(k * size) + j];
        }
        result[(i * size) + j] = sum;
      }
    }
  });
  return result;
}

std::vector<double> StrassenTbb::StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b,
                                                  int size) {
  if (size <= 64) {
    return StandardMultiply(a, b, size);
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

  tbb::parallel_invoke(
      [&] { SplitMatrix(a, a11, 0, 0, size); }, [&] { SplitMatrix(a, a12, 0, half_size, size); },
      [&] { SplitMatrix(a, a21, half_size, 0, size); }, [&] { SplitMatrix(a, a22, half_size, half_size, size); },
      [&] { SplitMatrix(b, b11, 0, 0, size); }, [&] { SplitMatrix(b, b12, 0, half_size, size); },
      [&] { SplitMatrix(b, b21, half_size, 0, size); }, [&] { SplitMatrix(b, b22, half_size, half_size, size); });

  std::vector<double> s1(half_size * half_size), s2(half_size * half_size), s3(half_size * half_size),
      s4(half_size * half_size), s5(half_size * half_size), s6(half_size * half_size), s7(half_size * half_size),
      s8(half_size * half_size), s9(half_size * half_size), s10(half_size * half_size);

  tbb::parallel_invoke(
      [&] { s1 = AddMatrices(a11, a22, half_size); }, [&] { s2 = AddMatrices(b11, b22, half_size); },
      [&] { s3 = AddMatrices(a21, a22, half_size); }, [&] { s4 = SubtractMatrices(b12, b22, half_size); },
      [&] { s5 = SubtractMatrices(b21, b11, half_size); }, [&] { s6 = AddMatrices(a11, a12, half_size); },
      [&] { s7 = SubtractMatrices(a21, a11, half_size); }, [&] { s8 = AddMatrices(b11, b12, half_size); },
      [&] { s9 = SubtractMatrices(a12, a22, half_size); }, [&] { s10 = AddMatrices(b21, b22, half_size); });

  std::vector<double> p1;
  std::vector<double> p2;
  std::vector<double> p3;
  std::vector<double> p4;
  std::vector<double> p5;
  std::vector<double> p6;
  std::vector<double> p7;
  tbb::parallel_invoke(
      [&] {
        p1 = StrassenMultiply(s1, s2, half_size);
        p2 = StrassenMultiply(s3, b11, half_size);
      },
      [&] {
        p3 = StrassenMultiply(a11, s4, half_size);
        p4 = StrassenMultiply(a22, s5, half_size);
      },
      [&] {
        p5 = StrassenMultiply(s6, b22, half_size);
        p6 = StrassenMultiply(s7, s8, half_size);
      },
      [&] { p7 = StrassenMultiply(s9, s10, half_size); });

  std::vector<double> c11(half_size * half_size), c12(half_size * half_size), c21(half_size * half_size),
      c22(half_size * half_size);

  tbb::parallel_invoke(
      [&] { c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size); },
      [&] { c12 = AddMatrices(p3, p5, half_size); }, [&] { c21 = AddMatrices(p2, p4, half_size); },
      [&] { c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size); });

  std::vector<double> result(size * size);
  tbb::parallel_invoke([&] { MergeMatrix(result, c11, 0, 0, size); },
                       [&] { MergeMatrix(result, c12, 0, half_size, size); },
                       [&] { MergeMatrix(result, c21, half_size, 0, size); },
                       [&] { MergeMatrix(result, c22, half_size, half_size, size); });

  return result;
}

void StrassenTbb::SplitMatrix(const std::vector<double>& parent, std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(parent.begin() + (row_start + i) * parent_size + col_start,
                      parent.begin() + (row_start + i) * parent_size + col_start + child_size,
                      child.begin() + i * child_size);
  }
}

void StrassenTbb::MergeMatrix(std::vector<double>& parent, const std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(child.begin() + i * child_size, child.begin() + (i + 1) * child_size,
                      parent.begin() + (row_start + i) * parent_size + col_start);
  }
}

std::vector<double> StrassenTbb::PadMatrixToPowerOfTwo(const std::vector<double>& matrix, int original_size) {
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

std::vector<double> StrassenTbb::TrimMatrixToOriginalSize(const std::vector<double>& matrix, int original_size,
                                                          int padded_size) {
  std::vector<double> trimmed_matrix(original_size * original_size);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * padded_size, matrix.begin() + i * padded_size + original_size,
                      trimmed_matrix.begin() + i * original_size);
  }
  return trimmed_matrix;
}

}  // namespace nasedkin_e_strassen_algorithm_tbb