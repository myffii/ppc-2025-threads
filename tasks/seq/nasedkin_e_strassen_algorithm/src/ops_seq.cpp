#include <cmath>
#include <cstddef>
#include <vector>

#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::PreProcessingImpl() {
  // Инициализация входных данных
  unsigned int input_size = taskData->inputs_count[0];
  auto* in_ptr_A = reinterpret_cast<int*>(taskData->inputs[0]);  // Первая матрица
  auto* in_ptr_B = reinterpret_cast<int*>(taskData->inputs[1]);  // Вторая матрица

  // Преобразуем входные данные в матрицы
  matrix_size_ = static_cast<int>(std::sqrt(input_size));
  input_matrix_A_.resize(matrix_size_, std::vector<int>(matrix_size_));
  input_matrix_B_.resize(matrix_size_, std::vector<int>(matrix_size_));

  for (int i = 0; i < matrix_size_; ++i) {
    for (int j = 0; j < matrix_size_; ++j) {
      input_matrix_A_[i][j] = in_ptr_A[i * matrix_size_ + j];
      input_matrix_B_[i][j] = in_ptr_B[i * matrix_size_ + j];
    }
  }

  // Инициализация выходной матрицы
  output_matrix_.resize(matrix_size_, std::vector<int>(matrix_size_, 0));
  return true;
}

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::ValidationImpl() {
  // Проверка корректности размеров входных данных
  unsigned int input_size_A = taskData->inputs_count[0];
  unsigned int input_size_B = taskData->inputs_count[1];
  unsigned int output_size = taskData->outputs_count[0];

  // Проверка, что входные матрицы квадратные и их размеры совпадают
  int size_A = static_cast<int>(std::sqrt(input_size_A));
  int size_B = static_cast<int>(std::sqrt(input_size_B));
  int size_output = static_cast<int>(std::sqrt(output_size));

  return (size_A == size_B) && (size_A == size_output);
}

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::RunImpl() {
  // Выполнение алгоритма Штрассена
  output_matrix_ = strassenMultiply(input_matrix_A_, input_matrix_B_);
  return true;
}

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::PostProcessingImpl() {
  // Сохранение результата в выходные данные
  auto* out_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  for (int i = 0; i < matrix_size_; ++i) {
    for (int j = 0; j < matrix_size_; ++j) {
      out_ptr[i * matrix_size_ + j] = output_matrix_[i][j];
    }
  }
  return true;
}

vector<vector<int>> nasedkin_e_strassen_algorithm_seq::StrassenSequential::addMatrices(const vector<vector<int>>& A,
                                                                                       const vector<vector<int>>& B) {
  int n = A.size();
  vector<vector<int>> result(n, vector<int>(n));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      result[i][j] = A[i][j] + B[i][j];
    }
  }
  return result;
}

// Функция для вычитания двух матриц
vector<vector<int>> nasedkin_e_strassen_algorithm_seq::StrassenSequential::subtractMatrices(
    const vector<vector<int>>& A, const vector<vector<int>>& B) {
  int n = A.size();
  vector<vector<int>> result(n, vector<int>(n));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      result[i][j] = A[i][j] - B[i][j];
    }
  }
  return result;
}

// Функция для разделения матрицы на 4 подматрицы
void nasedkin_e_strassen_algorithm_seq::StrassenSequential::splitMatrix(const vector<vector<int>>& parent,
                                                                        vector<vector<int>>& child, int rowStart,
                                                                        int colStart) {
  for (int i = 0; i < child.size(); i++) {
    for (int j = 0; j < child.size(); j++) {
      child[i][j] = parent[rowStart + i][colStart + j];
    }
  }
}

// Функция для слияния 4 подматриц в одну матрицу
void nasedkin_e_strassen_algorithm_seq::StrassenSequential::mergeMatrix(vector<vector<int>>& parent,
                                                                        const vector<vector<int>>& child, int rowStart,
                                                                        int colStart) {
  for (int i = 0; i < child.size(); i++) {
    for (int j = 0; j < child.size(); j++) {
      parent[rowStart + i][colStart + j] = child[i][j];
    }
  }
}

// Стандартное умножение матриц (базовый случай)
vector<vector<int>> nasedkin_e_strassen_algorithm_seq::StrassenSequential::standardMultiply(
    const vector<vector<int>>& A, const vector<vector<int>>& B) {
  int n = A.size();
  vector<vector<int>> result(n, vector<int>(n, 0));
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
vector<vector<int>> nasedkin_e_strassen_algorithm_seq::StrassenSequential::strassenMultiply(
    const vector<vector<int>>& A, const vector<vector<int>>& B) {
  int n = A.size();

  // Базовый случай: если матрица маленькая, используем стандартное умножение
  if (n <= 2) {
    return standardMultiply(A, B);
  }

  // Разделение матриц на подматрицы
  int halfSize = n / 2;
  vector<vector<int>> A11(halfSize, vector<int>(halfSize));
  vector<vector<int>> A12(halfSize, vector<int>(halfSize));
  vector<vector<int>> A21(halfSize, vector<int>(halfSize));
  vector<vector<int>> A22(halfSize, vector<int>(halfSize));
  splitMatrix(A, A11, 0, 0);
  splitMatrix(A, A12, 0, halfSize);
  splitMatrix(A, A21, halfSize, 0);
  splitMatrix(A, A22, halfSize, halfSize);

  vector<vector<int>> B11(halfSize, vector<int>(halfSize));
  vector<vector<int>> B12(halfSize, vector<int>(halfSize));
  vector<vector<int>> B21(halfSize, vector<int>(halfSize));
  vector<vector<int>> B22(halfSize, vector<int>(halfSize));
  splitMatrix(B, B11, 0, 0);
  splitMatrix(B, B12, 0, halfSize);
  splitMatrix(B, B21, halfSize, 0);
  splitMatrix(B, B22, halfSize, halfSize);

  // Вычисление промежуточных матриц P1-P7
  vector<vector<int>> P1 = strassenMultiply(addMatrices(A11, A22), addMatrices(B11, B22));
  vector<vector<int>> P2 = strassenMultiply(addMatrices(A21, A22), B11);
  vector<vector<int>> P3 = strassenMultiply(A11, subtractMatrices(B12, B22));
  vector<vector<int>> P4 = strassenMultiply(A22, subtractMatrices(B21, B11));
  vector<vector<int>> P5 = strassenMultiply(addMatrices(A11, A12), B22);
  vector<vector<int>> P6 = strassenMultiply(subtractMatrices(A21, A11), addMatrices(B11, B12));
  vector<vector<int>> P7 = strassenMultiply(subtractMatrices(A12, A22), addMatrices(B21, B22));

  // Вычисление результирующих подматриц C11, C12, C21, C22
  vector<vector<int>> C11 = addMatrices(subtractMatrices(addMatrices(P1, P4), P5), P7);
  vector<vector<int>> C12 = addMatrices(P3, P5);
  vector<vector<int>> C21 = addMatrices(P2, P4);
  vector<vector<int>> C22 = addMatrices(subtractMatrices(addMatrices(P1, P3), P2), P6);

  // Слияние подматриц в результирующую матрицу
  vector<vector<int>> result(n, vector<int>(n));
  mergeMatrix(result, C11, 0, 0);
  mergeMatrix(result, C12, 0, halfSize);
  mergeMatrix(result, C21, halfSize, 0);
  mergeMatrix(result, C22, halfSize, halfSize);

  return result;
}
