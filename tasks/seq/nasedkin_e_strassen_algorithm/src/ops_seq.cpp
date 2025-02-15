#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>  // Для size_t
#include <vector>

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::PreProcessingImpl() {
  // Инициализация входных данных
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr_a = reinterpret_cast<int*>(task_data->inputs[0]);  // Первая матрица
  auto* in_ptr_b = reinterpret_cast<int*>(task_data->inputs[1]);  // Вторая матрица

  // Преобразуем входные данные в матрицы
  matrix_size_ = static_cast<int>(std::sqrt(input_size));
  input_matrix_a_.resize(matrix_size_, std::vector<int>(matrix_size_));
  input_matrix_b_.resize(matrix_size_, std::vector<int>(matrix_size_));

  for (int i = 0; i < matrix_size_; ++i) {
    for (int j = 0; j < matrix_size_; ++j) {
      input_matrix_a_[i][j] = in_ptr_a[i * matrix_size_ + j];
      input_matrix_b_[i][j] = in_ptr_b[i * matrix_size_ + j];
    }
  }

  // Инициализация выходной матрицы
  output_matrix_.resize(matrix_size_, std::vector<int>(matrix_size_, 0));
  return true;
}

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::ValidationImpl() {
  // Проверка корректности размеров входных данных
  unsigned int input_size_a = task_data->inputs_count[0];
  unsigned int input_size_b = task_data->inputs_count[1];
  unsigned int output_size = task_data->outputs_count[0];

  // Проверка, что входные матрицы квадратные и их размеры совпадают
  int size_a = static_cast<int>(std::sqrt(input_size_a));
  int size_b = static_cast<int>(std::sqrt(input_size_b));
  int size_output = static_cast<int>(std::sqrt(output_size));

  return (size_a == size_b) && (size_a == size_output);
}

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::RunImpl() {
  // Выполнение алгоритма Штрассена
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_);
  return true;
}

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::PostProcessingImpl() {
  // Сохранение результата в выходные данные
  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  for (int i = 0; i < matrix_size_; ++i) {
    for (int j = 0; j < matrix_size_; ++j) {
      out_ptr[i * matrix_size_ + j] = output_matrix_[i][j];
    }
  }
  return true;
}

std::vector<std::vector<int>> nasedkin_e_strassen_algorithm_seq::StrassenSequential::AddMatrices(
    const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b) {
  int n = static_cast<int>(a.size());
  std::vector<std::vector<int>> result(n, std::vector<int>(n));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      result[i][j] = a[i][j] + b[i][j];
    }
  }
  return result;
}

std::vector<std::vector<int>> nasedkin_e_strassen_algorithm_seq::StrassenSequential::SubtractMatrices(
    const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b) {
  int n = static_cast<int>(a.size());
  std::vector<std::vector<int>> result(n, std::vector<int>(n));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      result[i][j] = a[i][j] - b[i][j];
    }
  }
  return result;
}

void nasedkin_e_strassen_algorithm_seq::StrassenSequential::SplitMatrix(const std::vector<std::vector<int>>& parent,
                                                                        std::vector<std::vector<int>>& child,
                                                                        int row_start, int col_start) {
  for (size_t i = 0; i < child.size(); i++) {
    for (size_t j = 0; j < child.size(); j++) {
      child[i][j] = parent[row_start + i][col_start + j];
    }
  }
}

void nasedkin_e_strassen_algorithm_seq::StrassenSequential::MergeMatrix(std::vector<std::vector<int>>& parent,
                                                                        const std::vector<std::vector<int>>& child,
                                                                        int row_start, int col_start) {
  for (size_t i = 0; i < child.size(); i++) {
    for (size_t j = 0; j < child.size(); j++) {
      parent[row_start + i][col_start + j] = child[i][j];
    }
  }
}

std::vector<std::vector<int>> nasedkin_e_strassen_algorithm_seq::StrassenSequential::StandardMultiply(
    const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b) {
  int n = static_cast<int>(a.size());
  std::vector<std::vector<int>> result(n, std::vector<int>(n, 0));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return result;
}

std::vector<std::vector<int>> nasedkin_e_strassen_algorithm_seq::StrassenSequential::StrassenMultiply(
    const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b) {
  int n = static_cast<int>(a.size());

  // Базовый случай: если матрица маленькая, используем стандартное умножение
  if (n <= 2) {
    return StandardMultiply(a, b);
  }

  // Разделение матриц на подматрицы
  int half_size = n / 2;
  std::vector<std::vector<int>> a11(half_size, std::vector<int>(half_size));
  std::vector<std::vector<int>> a12(half_size, std::vector<int>(half_size));
  std::vector<std::vector<int>> a21(half_size, std::vector<int>(half_size));
  std::vector<std::vector<int>> a22(half_size, std::vector<int>(half_size));
  SplitMatrix(a, a11, 0, 0);
  SplitMatrix(a, a12, 0, half_size);
  SplitMatrix(a, a21, half_size, 0);
  SplitMatrix(a, a22, half_size, half_size);

  std::vector<std::vector<int>> b11(half_size, std::vector<int>(half_size));
  std::vector<std::vector<int>> b12(half_size, std::vector<int>(half_size));
  std::vector<std::vector<int>> b21(half_size, std::vector<int>(half_size));
  std::vector<std::vector<int>> b22(half_size, std::vector<int>(half_size));
  SplitMatrix(b, b11, 0, 0);
  SplitMatrix(b, b12, 0, half_size);
  SplitMatrix(b, b21, half_size, 0);
  SplitMatrix(b, b22, half_size, half_size);

  // Вычисление промежуточных матриц P1-P7
  std::vector<std::vector<int>> p1 = StrassenMultiply(AddMatrices(a11, a22), AddMatrices(b11, b22));
  std::vector<std::vector<int>> p2 = StrassenMultiply(AddMatrices(a21, a22), b11);
  std::vector<std::vector<int>> p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22));
  std::vector<std::vector<int>> p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11));
  std::vector<std::vector<int>> p5 = StrassenMultiply(AddMatrices(a11, a12), b22);
  std::vector<std::vector<int>> p6 = StrassenMultiply(SubtractMatrices(a21, a11), AddMatrices(b11, b12));
  std::vector<std::vector<int>> p7 = StrassenMultiply(SubtractMatrices(a12, a22), AddMatrices(b21, b22));

  // Вычисление результирующих подматриц C11, C12, C21, C22
  std::vector<std::vector<int>> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4), p5), p7);
  std::vector<std::vector<int>> c12 = AddMatrices(p3, p5);
  std::vector<std::vector<int>> c21 = AddMatrices(p2, p4);
  std::vector<std::vector<int>> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3), p2), p6);

  // Слияние подматриц в результирующую матрицу
  std::vector<std::vector<int>> result(n, std::vector<int>(n));
  MergeMatrix(result, c11, 0, 0);
  MergeMatrix(result, c12, 0, half_size);
  MergeMatrix(result, c21, half_size, 0);
  MergeMatrix(result, c22, half_size, half_size);

  return result;
}