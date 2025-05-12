#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace nasedkin_e_strassen_algorithm_all {

bool StrassenAll::PreProcessingImpl() {
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

bool StrassenAll::ValidationImpl() {
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

bool StrassenAll::RunImpl() {
  boost::mpi::communicator world;
  int rank = world.rank();

  // Синхронизируем входные матрицы
  boost::mpi::broadcast(world, input_matrix_a_, 0);
  boost::mpi::broadcast(world, input_matrix_b_, 0);

  // Отладочный вывод: входные данные после синхронизации
  std::cout << "[DEBUG] Process " << rank << ": After input synchronization: "
            << "input_matrix_a_[0] = " << (input_matrix_a_.empty() ? 0.0 : input_matrix_a_[0])
            << ", input_matrix_b_[0] = " << (input_matrix_b_.empty() ? 0.0 : input_matrix_b_[0]) << std::endl;

  int num_threads = std::min(16, ppc::util::GetPPCNumThreads());
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_, num_threads);
  return true;
}

bool StrassenAll::PostProcessingImpl() {
  if (original_size_ != matrix_size_) {
    output_matrix_ = TrimMatrixToOriginalSize(output_matrix_, original_size_, matrix_size_);
  }

  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(output_matrix_, out_ptr);
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
  boost::mpi::communicator world;
  int rank = world.rank();
  int world_size = world.size();

  std::cout << "[DEBUG] Process " << rank << ": Starting StrassenMultiply with size = " << size
            << ", a[0] = " << (a.empty() ? 0.0 : a[0]) << ", b[0] = " << (b.empty() ? 0.0 : b[0])
            << ", num_threads = " << num_threads << std::endl;

  // Базовый случай: переход к стандартному умножению для малых матриц или одного процесса
  if (size <= 32 || world_size <= 1) {
    std::cout << "[DEBUG] Process " << rank << ": Switching to StandardMultiply for size = " << size << std::endl;
    return StandardMultiply(a, b, size);
  }

  // Проверка корректности размеров входных матриц
  if (static_cast<int>(a.size()) != size * size || static_cast<int>(b.size()) != size * size) {
    std::cerr << "[ERROR] Process " << rank << ": Invalid matrix dimensions: a.size = " << a.size()
              << ", b.size = " << b.size() << ", expected = " << size * size << std::endl;
    return {};
  }

  int half_size = size / 2;
  int half_size_squared = half_size * half_size;

  // Разделение матриц на подматрицы
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

  std::cout << "[DEBUG] Process " << rank << ": Submatrices created: a11.size = " << a11.size()
            << ", b11.size = " << b11.size() << std::endl;

  // Инициализация промежуточных результатов p1–p7
  std::vector<double> p1(half_size_squared, 0.0);
  std::vector<double> p2(half_size_squared, 0.0);
  std::vector<double> p3(half_size_squared, 0.0);
  std::vector<double> p4(half_size_squared, 0.0);
  std::vector<double> p5(half_size_squared, 0.0);
  std::vector<double> p6(half_size_squared, 0.0);
  std::vector<double> p7(half_size_squared, 0.0);

  // Создание задач для алгоритма Штрассена
  std::vector<std::function<void()>> tasks;
  tasks.emplace_back([&]() {
    p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, num_threads);
  });
  tasks.emplace_back([&]() { p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads); });
  tasks.emplace_back(
      [&]() { p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads); });
  tasks.emplace_back(
      [&]() { p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads); });
  tasks.emplace_back([&]() { p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads); });
  tasks.emplace_back([&]() {
    p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                          num_threads);
  });
  tasks.emplace_back([&]() {
    p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                          num_threads);
  });

  // Распределение задач между процессами
  if (rank < static_cast<int>(tasks.size()) && rank < world_size) {
    std::cout << "[DEBUG] Process " << rank << ": Executing task p" << (rank + 1) << std::endl;
    tasks[rank]();
  }

  // Синхронизация результатов
  std::cout << "[DEBUG] Process " << rank << ": Before broadcast" << std::endl;
  world.barrier();
  boost::mpi::broadcast(world, p1, 0);
  boost::mpi::broadcast(world, p2, 0);
  boost::mpi::broadcast(world, p3, 0);
  boost::mpi::broadcast(world, p4, 0);
  boost::mpi::broadcast(world, p5, 0);
  boost::mpi::broadcast(world, p6, 0);
  boost::mpi::broadcast(world, p7, 0);
  world.barrier();
  std::cout << "[DEBUG] Process " << rank << ": After broadcast" << std::endl;

  // Выполнение оставшихся задач последовательно на процессе 0
  if (rank == 0) {
    for (std::size_t i = world_size; i < tasks.size(); ++i) {
      std::cout << "[DEBUG] Process " << rank << ": Executing remaining task p" << (i + 1) << std::endl;
      tasks[i]();
    }
  }

  // Проверка размеров промежуточных результатов
  if (p1.size() != static_cast<std::size_t>(half_size_squared) ||
      p2.size() != static_cast<std::size_t>(half_size_squared) ||
      p3.size() != static_cast<std::size_t>(half_size_squared) ||
      p4.size() != static_cast<std::size_t>(half_size_squared) ||
      p5.size() != static_cast<std::size_t>(half_size_squared) ||
      p6.size() != static_cast<std::size_t>(half_size_squared) ||
      p7.size() != static_cast<std::size_t>(half_size_squared)) {
    std::cerr << "[ERROR] Process " << rank << ": Invalid size of intermediate results" << std::endl;
    return {};
  }

  // Формирование результирующих подматриц
  std::vector<double> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  std::vector<double> c12 = AddMatrices(p3, p5, half_size);
  std::vector<double> c21 = AddMatrices(p2, p4, half_size);
  std::vector<double> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

  // Проверка размеров результирующих подматриц
  if (c11.size() != static_cast<std::size_t>(half_size_squared) ||
      c12.size() != static_cast<std::size_t>(half_size_squared) ||
      c21.size() != static_cast<std::size_t>(half_size_squared) ||
      c22.size() != static_cast<std::size_t>(half_size_squared)) {
    std::cerr << "[ERROR] Process " << rank << ": Invalid size of result submatrices" << std::endl;
    return {};
  }

  // Сборка итоговой матрицы
  std::vector<double> result(size * size);
  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);

  std::cout << "[DEBUG] Process " << rank << ": StrassenMultiply completed with result.size = " << result.size()
            << ", result[0] = " << (result.empty() ? 0.0 : result[0]) << std::endl;

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