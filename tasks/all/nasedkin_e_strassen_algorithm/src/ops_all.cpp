#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <cmath>
#include <cstddef>
#include <functional>
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
  // Инициализация MPI, если еще не инициализирован
  boost::mpi::environment* env = nullptr;
  if (!boost::mpi::environment::initialized()) {
    static int argc = 0;
    static char** argv = nullptr;
    env = new boost::mpi::environment(argc, argv);
  }

  boost::mpi::communicator world;
  int rank = world.rank();
  int world_size = world.size();

  // Базовый случай или слишком маленькая матрица для распараллеливания
  if (size <= 32 || world_size <= 1) {
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

  // На всех процессах разделяем матрицы на подматрицы
  SplitMatrix(a, a11, 0, 0, size);
  SplitMatrix(a, a12, 0, half_size, size);
  SplitMatrix(a, a21, half_size, 0, size);
  SplitMatrix(a, a22, half_size, half_size, size);

  SplitMatrix(b, b11, 0, 0, size);
  SplitMatrix(b, b12, 0, half_size, size);
  SplitMatrix(b, b21, half_size, 0, size);
  SplitMatrix(b, b22, half_size, half_size, size);

  std::vector<double> p1;
  std::vector<double> p2;
  std::vector<double> p3;
  std::vector<double> p4;
  std::vector<double> p5;
  std::vector<double> p6;
  std::vector<double> p7;

  // Изменим распределение вычислений между процессами с учетом фактического количества процессов
  if (world_size < 7) {
    if (rank == 0) {
      p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, num_threads);
      p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads);
      p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads);
      p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads);
      p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads);
      p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                            num_threads);
      p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                            num_threads);
    }
  } else {
    int task_id = rank % 7;
    switch (task_id) {
      case 0: {
        p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size,
                              num_threads);
        boost::mpi::broadcast(world, p1, 0);
        break;
      }
      case 1: {
        p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads);
        boost::mpi::broadcast(world, p2, 1);
        break;
      }
      case 2: {
        p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads);
        boost::mpi::broadcast(world, p3, 2);
        break;
      }
      case 3: {
        p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads);
        boost::mpi::broadcast(world, p4, 3);
        break;
      }
      case 4: {
        p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads);
        boost::mpi::broadcast(world, p5, 4);
        break;
      }
      case 5: {
        p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                              num_threads);
        boost::mpi::broadcast(world, p6, 5);
        break;
      }
      case 6: {
        p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                              num_threads);
        boost::mpi::broadcast(world, p7, 6);
        break;
      }
      default: {
        break;
      }
    }

    if (world_size > 0 && task_id != 0) {
      boost::mpi::broadcast(world, p1, 0);
    }
    if (world_size > 1 && task_id != 1) {
      boost::mpi::broadcast(world, p2, 1);
    }
    if (world_size > 2 && task_id != 2) {
      boost::mpi::broadcast(world, p3, 2);
    }
    if (world_size > 3 && task_id != 3) {
      boost::mpi::broadcast(world, p4, 3);
    }
    if (world_size > 4 && task_id != 4) {
      boost::mpi::broadcast(world, p5, 4);
    }
    if (world_size > 5 && task_id != 5) {
      boost::mpi::broadcast(world, p6, 5);
    }
    if (world_size > 6 && task_id != 6) {
      boost::mpi::broadcast(world, p7, 6);
    }
  }

  // Запускаем многопоточное выполнение задач, не выполненных через MPI
  std::vector<std::function<void()>> remaining_tasks;

  if (p1.empty()) {
    remaining_tasks.emplace_back([&]() {
      p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, num_threads);
    });
  }

  if (p2.empty()) {
    remaining_tasks.emplace_back(
        [&]() { p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads); });
  }

  if (p3.empty()) {
    remaining_tasks.emplace_back(
        [&]() { p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads); });
  }

  if (p4.empty()) {
    remaining_tasks.emplace_back(
        [&]() { p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads); });
  }

  if (p5.empty()) {
    remaining_tasks.emplace_back(
        [&]() { p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads); });
  }

  if (p6.empty()) {
    remaining_tasks.emplace_back([&]() {
      p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                            num_threads);
    });
  }

  if (p7.empty()) {
    remaining_tasks.emplace_back([&]() {
      p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                            num_threads);
    });
  }

  // Запускаем многопоточное выполнение оставшихся задач
  std::vector<std::thread> threads;
  threads.reserve(std::min(num_threads, static_cast<int>(remaining_tasks.size())));
  size_t task_index = 0;

  for (int i = 0; i < std::min(num_threads, static_cast<int>(remaining_tasks.size())); ++i) {
    if (task_index < remaining_tasks.size()) {
      threads.emplace_back(remaining_tasks[task_index]);
      ++task_index;
    }
  }

  while (task_index < remaining_tasks.size()) {
    remaining_tasks[task_index]();
    ++task_index;
  }

  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  // Комбинируем результаты для получения финальной матрицы
  std::vector<double> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  std::vector<double> c12 = AddMatrices(p3, p5, half_size);
  std::vector<double> c21 = AddMatrices(p2, p4, half_size);
  std::vector<double> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

  std::vector<double> result(size * size);
  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);

  // Очищаем созданный объект environment
  delete env;

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