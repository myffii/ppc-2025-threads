#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
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

  if (world_.rank() == 0) {
    std::ranges::copy(in_ptr_a, in_ptr_a + input_size, input_matrix_a_.begin());
    std::ranges::copy(in_ptr_b, in_ptr_b + input_size, input_matrix_b_.begin());
  }

  // Рассылка матриц всем процессам
  boost::mpi::broadcast(world_, input_matrix_a_.data(), input_matrix_a_.size(), 0);
  boost::mpi::broadcast(world_, input_matrix_b_.data(), input_matrix_b_.size(), 0);

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
  if (world_.rank() == 0) {
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

  std::vector<double> p1(half_size_squared, 0.0);
  std::vector<double> p2(half_size_squared, 0.0);
  std::vector<double> p3(half_size_squared, 0.0);
  std::vector<double> p4(half_size_squared, 0.0);
  std::vector<double> p5(half_size_squared, 0.0);
  std::vector<double> p6(half_size_squared, 0.0);
  std::vector<double> p7(half_size_squared, 0.0);

  // Распределяем задачи между процессами
  std::vector<std::function<void()>> tasks;
  tasks.reserve(7);
  if (world_.rank() == 0) {
    tasks.emplace_back([&]() {
      p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, num_threads);
    });
  }
  if (world_.rank() == 1 || world_.size() == 1) {
    tasks.emplace_back([&]() { p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads); });
  }
  if (world_.rank() == 2 || world_.size() <= 2) {
    tasks.emplace_back(
        [&]() { p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads); });
  }
  if (world_.rank() == 3 || world_.size() <= 3) {
    tasks.emplace_back(
        [&]() { p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads); });
  }
  if (world_.rank() == 4 || world_.size() <= 4) {
    tasks.emplace_back([&]() { p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads); });
  }
  if (world_.rank() == 5 || world_.size() <= 5) {
    tasks.emplace_back([&]() {
      p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                            num_threads);
    });
  }
  if (world_.rank() == 6 || world_.size() <= 6) {
    tasks.emplace_back([&]() {
      p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                            num_threads);
    });
  }

  // Выполняем задачи в потоках внутри процесса
  std::vector<std::thread> threads;
  threads.reserve(std::min(num_threads, static_cast<int>(tasks.size())));
  size_t task_index = 0;

  for (int i = 0; i < std::min(num_threads, static_cast<int>(tasks.size())); ++i) {
    if (task_index < tasks.size()) {
      threads.emplace_back(tasks[task_index]);
      ++task_index;
    }
  }

  while (task_index < tasks.size()) {
    tasks[task_index]();
    ++task_index;
  }

  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  // Собираем результаты от всех процессов
  std::vector<double> p1_global(half_size_squared, 0.0);
  std::vector<double> p2_global(half_size_squared, 0.0);
  std::vector<double> p3_global(half_size_squared, 0.0);
  std::vector<double> p4_global(half_size_squared, 0.0);
  std::vector<double> p5_global(half_size_squared, 0.0);
  std::vector<double> p6_global(half_size_squared, 0.0);
  std::vector<double> p7_global(half_size_squared, 0.0);

  boost::mpi::all_reduce(world_, p1.data(), p1.size(), p1_global.data(), std::plus<double>());
  boost::mpi::all_reduce(world_, p2.data(), p2.size(), p2_global.data(), std::plus<double>());
  boost::mpi::all_reduce(world_, p3.data(), p3.size(), p3_global.data(), std::plus<double>());
  boost::mpi::all_reduce(world_, p4.data(), p4.size(), p4_global.data(), std::plus<double>());
  boost::mpi::all_reduce(world_, p5.data(), p5.size(), p5_global.data(), std::plus<double>());
  boost::mpi::all_reduce(world_, p6.data(), p6.size(), p6_global.data(), std::plus<double>());
  boost::mpi::all_reduce(world_, p7.data(), p7.size(), p7_global.data(), std::plus<double>());

  // Вычисляем результирующие подматрицы
  std::vector<double> c11 = AddMatrices(
      SubtractMatrices(AddMatrices(p1_global, p4_global, half_size), p5_global, half_size), p7_global, half_size);
  std::vector<double> c12 = AddMatrices(p3_global, p5_global, half_size);
  std::vector<double> c21 = AddMatrices(p2_global, p4_global, half_size);
  std::vector<double> c22 = AddMatrices(
      SubtractMatrices(AddMatrices(p1_global, p3_global, half_size), p2_global, half_size), p6_global, half_size);

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