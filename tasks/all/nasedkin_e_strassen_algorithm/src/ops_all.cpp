#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
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
  boost::mpi::communicator world;
  int rank = world.rank();
  int num_processes = world.size();

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

  if (rank == 0) {
    SplitMatrix(a, a11, 0, 0, size);
    SplitMatrix(a, a12, 0, half_size, size);
    SplitMatrix(a, a21, half_size, 0, size);
    SplitMatrix(a, a22, half_size, half_size, size);
    SplitMatrix(b, b11, 0, 0, size);
    SplitMatrix(b, b12, 0, half_size, size);
    SplitMatrix(b, b21, half_size, 0, size);
    SplitMatrix(b, b22, half_size, half_size, size);
  }

  // Рассылка подматриц всем процессам
  boost::mpi::broadcast(world, a11, 0);
  boost::mpi::broadcast(world, a12, 0);
  boost::mpi::broadcast(world, a21, 0);
  boost::mpi::broadcast(world, a22, 0);
  boost::mpi::broadcast(world, b11, 0);
  boost::mpi::broadcast(world, b12, 0);
  boost::mpi::broadcast(world, b21, 0);
  boost::mpi::broadcast(world, b22, 0);

  std::vector<double> p1(half_size_squared);
  std::vector<double> p2(half_size_squared);
  std::vector<double> p3(half_size_squared);
  std::vector<double> p4(half_size_squared);
  std::vector<double> p5(half_size_squared);
  std::vector<double> p6(half_size_squared);
  std::vector<double> p7(half_size_squared);

  std::vector<std::function<void()>> tasks;
  tasks.reserve(7);
  if (rank == 0 || num_processes == 1) {
    tasks.emplace_back([&]() {
      p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, num_threads);
    });
  }
  if (rank == 1 || num_processes == 1) {
    tasks.emplace_back([&]() { p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads); });
  }
  if (rank == 2 || num_processes == 1) {
    tasks.emplace_back(
        [&]() { p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads); });
  }
  if (rank == 3 || num_processes == 1) {
    tasks.emplace_back(
        [&]() { p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads); });
  }
  if (rank == 4 || num_processes == 1) {
    tasks.emplace_back([&]() { p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads); });
  }
  if (rank == 5 || num_processes == 1) {
    tasks.emplace_back([&]() {
      p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                            num_threads);
    });
  }
  if (rank == 6 || num_processes == 1) {
    tasks.emplace_back([&]() {
      p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                            num_threads);
    });
  }

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

  // Сбор результатов p1-p7 на процессе с rank 0
  if (rank == 0 || num_processes == 1) {
    // Уже есть p1 если rank == 0 или num_processes == 1
  } else if (rank == 1) {
    boost::mpi::send(world, 0, 1, p2);
  } else if (rank == 2) {
    boost::mpi::send(world, 0, 2, p3);
  } else if (rank == 3) {
    boost::mpi::send(world, 0, 3, p4);
  } else if (rank == 4) {
    boost::mpi::send(world, 0, 4, p5);
  } else if (rank == 5) {
    boost::mpi::send(world, 0, 5, p6);
  } else if (rank == 6) {
    boost::mpi::send(world, 0, 6, p7);
  }

  if (rank == 0) {
    if (num_processes > 1) {
      boost::mpi::status s;
      if (num_processes > 1) s = world.recv(1, 1, p2);
      if (num_processes > 2) s = world.recv(2, 2, p3);
      if (num_processes > 3) s = world.recv(3, 3, p4);
      if (num_processes > 4) s = world.recv(4, 4, p5);
      if (num_processes > 5) s = world.recv(5, 5, p6);
      if (num_processes > 6) s = world.recv(6, 6, p7);
    }

    std::vector<double> c11 =
        AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
    std::vector<double> c12 = AddMatrices(p3, p5, half_size);
    std::vector<double> c21 = AddMatrices(p2, p4, half_size);
    std::vector<double> c22 =
        AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

    std::vector<double> result(size * size);
    MergeMatrix(result, c11, 0, 0, size);
    MergeMatrix(result, c12, 0, half_size, size);
    MergeMatrix(result, c21, half_size, 0, size);
    MergeMatrix(result, c22, half_size, half_size, size);

    return result;
  }

  return std::vector<double>();  // Процессы с rank != 0 возвращают пустой вектор
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