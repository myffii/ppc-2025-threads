#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <cmath>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace nasedkin_e_strassen_algorithm_all {

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

bool StrassenAll::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];

  if (world_.rank() == 0) {
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
  }

  // Broadcast matrix dimensions to all processes
  boost::mpi::broadcast(world_, matrix_size_, 0);
  boost::mpi::broadcast(world_, original_size_, 0);

  // Resize matrices on non-root processes
  if (world_.rank() != 0) {
    input_matrix_a_.resize(matrix_size_ * matrix_size_);
    input_matrix_b_.resize(matrix_size_ * matrix_size_);
    output_matrix_.resize(matrix_size_ * matrix_size_, 0.0);
  }

  // Broadcast input matrices to all processes
  boost::mpi::broadcast(world_, input_matrix_a_, 0);
  boost::mpi::broadcast(world_, input_matrix_b_, 0);

  return true;
}

bool StrassenAll::ValidationImpl() {
  if (world_.rank() == 0) {
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

  // Share validation result with all processes
  bool is_valid = false;
  if (world_.rank() == 0) {
    is_valid = true;  // Set to true since we passed the above checks
  }
  boost::mpi::broadcast(world_, is_valid, 0);

  return is_valid;
}

bool StrassenAll::RunImpl() {
  int num_threads = std::min(16, ppc::util::GetPPCNumThreads());
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_, num_threads);
  return true;
}

bool StrassenAll::PostProcessingImpl() {
  if (world_.rank() == 0) {
    if (original_size_ != matrix_size_) {
      output_matrix_ = TrimMatrixToOriginalSize(output_matrix_, original_size_, matrix_size_);
    }

    auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
    std::ranges::copy(output_matrix_, out_ptr);
  }
  return true;
}

std::vector<double> StrassenAll::AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::plus<>());
  return result;
}

std::vector<double> StrassenAll::SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b,
                                                  int size) {
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::minus<>());
  return result;
}

std::vector<double> StrassenAll::PadMatrixToPowerOfTwo(const std::vector<double>& matrix, int original_size) {
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

std::vector<double> StrassenAll::TrimMatrixToOriginalSize(const std::vector<double>& matrix, int original_size,
                                                          int padded_size) {
  std::vector<double> trimmed_matrix(original_size * original_size);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * padded_size, matrix.begin() + i * padded_size + original_size,
                      trimmed_matrix.begin() + i * original_size);
  }
  return trimmed_matrix;
}

std::vector<double> StrassenAll::StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b, int size,
                                                  int num_threads) {
  if (size <= 32) {
    return StandardMultiply(a, b, size);
  }

  int half_size = size / 2;
  int half_size_squared = half_size * half_size;

  std::vector<double> a11(half_size_squared);
  std::vector<double> a12(half_size_squared);
  std::vector<double> a21(half_size_squared);
  std::vector<double> a22(half_size_squared);
  std::vector<double> b11(half_size_squared);
  std::vector<double> b12(half_size_squared);
  std::vector<double> b21(half_size_squared);
  std::vector<double> b22(half_size_squared);

  SplitMatrix(a, a11, 0, 0, size);
  SplitMatrix(a, a12, 0, half_size, size);
  SplitMatrix(a, a21, half_size, 0, size);
  SplitMatrix(a, a22, half_size, half_size, size);

  SplitMatrix(b, b11, 0, 0, size);
  SplitMatrix(b, b12, 0, half_size, size);
  SplitMatrix(b, b21, half_size, 0, size);
  SplitMatrix(b, b22, half_size, half_size, size);

  std::vector<double> p1, p2, p3, p4, p5, p6, p7;

  // Распределение задач между MPI-процессами
  int world_size = world_.size();
  int my_rank = world_.rank();

  // Определяем, какие умножения будут выполнены на каждом процессе
  // Используем гибридный подход: распределяем 7 умножений по MPI-процессам
  // каждый процесс использует внутри себя многопоточность через std::thread

  // Процессор с рангом r выполняет умножения с индексами r, r+world_size, r+2*world_size, ...
  std::vector<int> my_multiplications;
  for (int i = my_rank; i < 7; i += world_size) {
    my_multiplications.push_back(i);
  }

  // Локальные вычисления для каждого процесса
  for (int mult_idx : my_multiplications) {
    switch (mult_idx) {
      case 0:
        p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size,
                              num_threads);
        break;
      case 1:
        p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads);
        break;
      case 2:
        p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads);
        break;
      case 3:
        p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads);
        break;
      case 4:
        p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads);
        break;
      case 5:
        p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                              num_threads);
        break;
      case 6:
        p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                              num_threads);
        break;
    }
  }

  // Если размер не был вычислен (другой процесс делал эту работу), инициализируем пустым вектором нужного размера
  if (p1.empty() && half_size > 0) p1.resize(half_size_squared);
  if (p2.empty() && half_size > 0) p2.resize(half_size_squared);
  if (p3.empty() && half_size > 0) p3.resize(half_size_squared);
  if (p4.empty() && half_size > 0) p4.resize(half_size_squared);
  if (p5.empty() && half_size > 0) p5.resize(half_size_squared);
  if (p6.empty() && half_size > 0) p6.resize(half_size_squared);
  if (p7.empty() && half_size > 0) p7.resize(half_size_squared);

  // Собираем результаты со всех процессов
  for (int i = 0; i < 7; ++i) {
    int src_rank = i % world_size;  // Процесс, который вычислил результат

    // Выбираем соответствующий вектор и синхронизируем его
    std::vector<double>* p_vector = nullptr;
    switch (i) {
      case 0:
        p_vector = &p1;
        break;
      case 1:
        p_vector = &p2;
        break;
      case 2:
        p_vector = &p3;
        break;
      case 3:
        p_vector = &p4;
        break;
      case 4:
        p_vector = &p5;
        break;
      case 5:
        p_vector = &p6;
        break;
      case 6:
        p_vector = &p7;
        break;
    }

    if (p_vector) {
      // Рассылаем данные от процесса-источника всем остальным
      boost::mpi::broadcast(world_, *p_vector, src_rank);
    }
  }

  // Все процессы теперь имеют полные данные для расчета
  std::vector<double> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  std::vector<double> c12 = AddMatrices(p3, p5, half_size);
  std::vector<double> c21 = AddMatrices(p2, p4, half_size);
  std::vector<double> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

  std::vector<double> result(size * size);
  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);

  return result;
}

void StrassenAll::SplitMatrix(const std::vector<double>& parent, std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(parent.begin() + (row_start + i) * parent_size + col_start,
                      parent.begin() + (row_start + i) * parent_size + col_start + child_size,
                      child.begin() + i * child_size);
  }
}

void StrassenAll::MergeMatrix(std::vector<double>& parent, const std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(child.begin() + i * child_size, child.begin() + (i + 1) * child_size,
                      parent.begin() + (row_start + i) * parent_size + col_start);
  }
}

}  // namespace nasedkin_e_strassen_algorithm_all