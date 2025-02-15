#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"

#include <cmath>
#include <vector>

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::PreProcessingImpl() {
  // Инициализация входных данных
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr_A = reinterpret_cast<int*>(task_data->inputs[0]);  // Первая матрица
  auto* in_ptr_B = reinterpret_cast<int*>(task_data->inputs[1]);  // Вторая матрица

  // Преобразуем входные данные в матрицы
  matrix_size_ = static_cast<int>(std::sqrt(input_size));
  input_matrix_a_.resize(matrix_size_, std::vector<int>(matrix_size_));
  input_matrix_b_.resize(matrix_size_, std::vector<int>(matrix_size_));

  for (int i = 0; i < matrix_size_; ++i) {
    for (int j = 0; j < matrix_size_; ++j) {
      input_matrix_a_[i][j] = in_ptr_A[i * matrix_size_ + j];
      input_matrix_b_[i][j] = in_ptr_B[i * matrix_size_ + j];
    }
  }

  // Инициализация выходной матрицы
  output_matrix_.resize(matrix_size_, std::vector<int>(matrix_size_, 0));
  return true;
}

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::ValidationImpl() {
  // Проверка корректности размеров входных данных
  unsigned int input_size_A = task_data->inputs_count[0];
  unsigned int input_size_B = task_data->inputs_count[1];
  unsigned int output_size = task_data->outputs_count[0];

  // Проверка, что входные матрицы квадратные и их размеры совпадают
  int size_A = static_cast<int>(std::sqrt(input_size_A));
  int size_B = static_cast<int>(std::sqrt(input_size_B));
  int size_output = static_cast<int>(std::sqrt(output_size));

  return (size_A == size_B) && (size_A == size_output);
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
    const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B) {
  int n = A.size();
  std::vector<std::vector<int>> result(n, std::vector<int>(n));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      result[i][j] = A[i][j] + B[i][j];
    }
  }
  return result;
}

// Функция для вычитания двух матриц
std::vector<std::vector<int>> nasedkin_e_strassen_algorithm_seq::StrassenSequential::SubtractMatrices(
    const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B) {
  int n = A.size();
  std::vector<std::vector<int>> result(n, std::vector<int>(n));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      result[i][j] = A[i][j] - B[i][j];
    }
  }
  return result;
}

// Функция для разделения матрицы на 4 подматрицы
void nasedkin_e_strassen_algorithm_seq::StrassenSequential::SplitMatrix(const std::vector<std::vector<int>>& parent,
                                                                        std::vector<std::vector<int>>& child,
                                                                        int rowStart, int colStart) {
  for (int i = 0; i < child.size(); i++) {
    for (int j = 0; j < child.size(); j++) {
      child[i][j] = parent[rowStart + i][colStart + j];
    }
  }
}

// Функция для слияния 4 подматриц в одну матрицу
void nasedkin_e_strassen_algorithm_seq::StrassenSequential::MergeMatrix(std::vector<std::vector<int>>& parent,
                                                                        const std::vector<std::vector<int>>& child,
                                                                        int rowStart, int colStart) {
  for (int i = 0; i < child.size(); i++) {
    for (int j = 0; j < child.size(); j++) {
      parent[rowStart + i][colStart + j] = child[i][j];
    }
  }
}

// Стандартное умножение матриц (базовый случай)
std::vector<std::vector<int>> nasedkin_e_strassen_algorithm_seq::StrassenSequential::StandardMultiply(
    const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B) {
  int n = A.size();
  std::vector<std::vector<int>> result(n, std::vector<int>(n, 0));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return result;
}

// Рекурсивная функция умножения матриц по алгоритму Штрассена
std::vector<std::vector<int>> nasedkin_e_strassen_algorithm_seq::StrassenSequential::StrassenMultiply(
    const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B) {
  int n = A.size();

  // Базовый случай: если матрица маленькая, используем стандартное умножение
  if (n <= 2) {
    return StandardMultiply(A, B);
  }

  // Разделение матриц на подматрицы
  int halfSize = n / 2;
  std::vector<std::vector<int>> A11(halfSize, std::vector<int>(halfSize));
  std::vector<std::vector<int>> A12(halfSize, std::vector<int>(halfSize));
  std::vector<std::vector<int>> A21(halfSize, std::vector<int>(halfSize));
  std::vector<std::vector<int>> A22(halfSize, std::vector<int>(halfSize));
  SplitMatrix(A, A11, 0, 0);
  SplitMatrix(A, A12, 0, halfSize);
  SplitMatrix(A, A21, halfSize, 0);
  SplitMatrix(A, A22, halfSize, halfSize);

  std::vector<std::vector<int>> B11(halfSize, std::vector<int>(halfSize));
  std::vector<std::vector<int>> B12(halfSize, std::vector<int>(halfSize));
  std::vector<std::vector<int>> B21(halfSize, std::vector<int>(halfSize));
  std::vector<std::vector<int>> B22(halfSize, std::vector<int>(halfSize));
  SplitMatrix(B, B11, 0, 0);
  SplitMatrix(B, B12, 0, halfSize);
  SplitMatrix(B, B21, halfSize, 0);
  SplitMatrix(B, B22, halfSize, halfSize);

  // Вычисление промежуточных матриц P1-P7
  std::vector<std::vector<int>> P1 = StrassenMultiply(AddMatrices(A11, A22), AddMatrices(B11, B22));
  std::vector<std::vector<int>> P2 = StrassenMultiply(AddMatrices(A21, A22), B11);
  std::vector<std::vector<int>> P3 = StrassenMultiply(A11, SubtractMatrices(B12, B22));
  std::vector<std::vector<int>> P4 = StrassenMultiply(A22, SubtractMatrices(B21, B11));
  std::vector<std::vector<int>> P5 = StrassenMultiply(AddMatrices(A11, A12), B22);
  std::vector<std::vector<int>> P6 = StrassenMultiply(SubtractMatrices(A21, A11), AddMatrices(B11, B12));
  std::vector<std::vector<int>> P7 = StrassenMultiply(SubtractMatrices(A12, A22), AddMatrices(B21, B22));

  // Вычисление результирующих подматриц C11, C12, C21, C22
  std::vector<std::vector<int>> C11 = AddMatrices(SubtractMatrices(AddMatrices(P1, P4), P5), P7);
  std::vector<std::vector<int>> C12 = AddMatrices(P3, P5);
  std::vector<std::vector<int>> C21 = AddMatrices(P2, P4);
  std::vector<std::vector<int>> C22 = AddMatrices(SubtractMatrices(AddMatrices(P1, P3), P2), P6);

  // Слияние подматриц в результирующую матрицу
  std::vector<std::vector<int>> result(n, std::vector<int>(n));
  MergeMatrix(result, C11, 0, 0);
  MergeMatrix(result, C12, 0, halfSize);
  MergeMatrix(result, C21, halfSize, 0);
  MergeMatrix(result, C22, halfSize, halfSize);

  return result;
}
