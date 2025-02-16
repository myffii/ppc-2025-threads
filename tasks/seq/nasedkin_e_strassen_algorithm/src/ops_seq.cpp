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
      input_matrix_a_[i][j] = in_ptr_a[(i * matrix_size_) + j];
      input_matrix_b_[i][j] = in_ptr_b[(i * matrix_size_) + j];
    }
  }

  // Проверяем, нужно ли дополнять матрицы до степени двойки
  if ((matrix_size_ & (matrix_size_ - 1)) != 0) {  // Если размер не является степенью двойки
    // Сохраняем исходный размер матрицы
    original_size_ = matrix_size_;

    // Дополняем матрицы до ближайшей степени двойки
    input_matrix_a_ = PadMatrixToPowerOfTwo(input_matrix_a_);
    input_matrix_b_ = PadMatrixToPowerOfTwo(input_matrix_b_);

    // Обновляем размер матрицы
    matrix_size_ = static_cast<int>(input_matrix_a_.size());
  } else {
    // Если размер уже степень двойки, исходный размер равен текущему
    original_size_ = matrix_size_;
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
  // Если матрицы были дополнены, обрезаем результат до исходного размера
  if (original_size_ != matrix_size_) {
    output_matrix_ = TrimMatrixToOriginalSize(output_matrix_, original_size_);
  }

  // Сохранение результата в выходные данные
  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  size_t output_size = original_size_ * original_size_;  // Используем исходный размер
  for (size_t i = 0; i < output_size; ++i) {
    out_ptr[i] = output_matrix_[i / original_size_][i % original_size_];
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

std::vector<std::vector<int>> nasedkin_e_strassen_algorithm_seq::StrassenSequential::PadMatrixToPowerOfTwo(
    const std::vector<std::vector<int>>& matrix) {
  // Определяем текущий размер матрицы
  size_t original_size = matrix.size();

  // Находим ближайшую степень двойки, которая больше или равна текущему размеру
  size_t new_size = 1;
  while (new_size < original_size) {
    new_size *= 2;
  }

  // Создаем новую матрицу размером new_size x new_size, заполненную нулями
  std::vector<std::vector<int>> padded_matrix(new_size, std::vector<int>(new_size, 0));

  // Копируем исходную матрицу в верхний левый угол новой матрицы
  for (size_t i = 0; i < original_size; ++i) {
    for (size_t j = 0; j < original_size; ++j) {
      padded_matrix[i][j] = matrix[i][j];
    }
  }

  return padded_matrix;
}

std::vector<std::vector<int>> nasedkin_e_strassen_algorithm_seq::StrassenSequential::TrimMatrixToOriginalSize(
    const std::vector<std::vector<int>>& matrix, size_t original_size) {
  std::vector<std::vector<int>> trimmed_matrix(original_size, std::vector<int>(original_size));

  for (size_t i = 0; i < original_size; ++i) {
    for (size_t j = 0; j < original_size; ++j) {
      trimmed_matrix[i][j] = matrix[i][j];
    }
  }

  return trimmed_matrix;
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
  if (n <= 32) {
    return StandardMultiply(a, b);
  }

  // Разделение матриц на подматрицы
  int half_size = n / 2;

  // Вспомогательные функции для работы с подматрицами через указатели
  auto get_submatrix = [](const std::vector<std::vector<int>>& matrix, int row_start, int col_start, int size) {
    std::vector<std::vector<int>> submatrix(size, std::vector<int>(size));
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        submatrix[i][j] = matrix[row_start + i][col_start + j];
      }
    }
    return submatrix;
  };

  auto add_submatrices = [](const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b, int size) {
    std::vector<std::vector<int>> result(size, std::vector<int>(size));
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        result[i][j] = a[i][j] + b[i][j];
      }
    }
    return result;
  };

  auto subtract_submatrices = [](const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b,
                                 int size) {
    std::vector<std::vector<int>> result(size, std::vector<int>(size));
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        result[i][j] = a[i][j] - b[i][j];
      }
    }
    return result;
  };

  // Получаем подматрицы через указатели
  auto a11 = get_submatrix(a, 0, 0, half_size);
  auto a12 = get_submatrix(a, 0, half_size, half_size);
  auto a21 = get_submatrix(a, half_size, 0, half_size);
  auto a22 = get_submatrix(a, half_size, half_size, half_size);

  auto b11 = get_submatrix(b, 0, 0, half_size);
  auto b12 = get_submatrix(b, 0, half_size, half_size);
  auto b21 = get_submatrix(b, half_size, 0, half_size);
  auto b22 = get_submatrix(b, half_size, half_size, half_size);

  // Вычисление промежуточных матриц P1-P7
  auto p1 = StrassenMultiply(add_submatrices(a11, a22, half_size), add_submatrices(b11, b22, half_size));
  auto p2 = StrassenMultiply(add_submatrices(a21, a22, half_size), b11);
  auto p3 = StrassenMultiply(a11, subtract_submatrices(b12, b22, half_size));
  auto p4 = StrassenMultiply(a22, subtract_submatrices(b21, b11, half_size));
  auto p5 = StrassenMultiply(add_submatrices(a11, a12, half_size), b22);
  auto p6 = StrassenMultiply(subtract_submatrices(a21, a11, half_size), add_submatrices(b11, b12, half_size));
  auto p7 = StrassenMultiply(subtract_submatrices(a12, a22, half_size), add_submatrices(b21, b22, half_size));

  // Вычисление результирующих подматриц C11, C12, C21, C22
  auto c11 = add_submatrices(subtract_submatrices(add_submatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  auto c12 = add_submatrices(p3, p5, half_size);
  auto c21 = add_submatrices(p2, p4, half_size);
  auto c22 = add_submatrices(subtract_submatrices(add_submatrices(p1, p3, half_size), p2, half_size), p6, half_size);

  // Слияние подматриц в результирующую матрицу
  std::vector<std::vector<int>> result(n, std::vector<int>(n));
  for (int i = 0; i < half_size; ++i) {
    for (int j = 0; j < half_size; ++j) {
      result[i][j] = c11[i][j];
      result[i][j + half_size] = c12[i][j];
      result[i + half_size][j] = c21[i][j];
      result[i + half_size][j + half_size] = c22[i][j];
    }
  }

  return result;
}