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
    std::cout << "[DEBUG] Padded matrices from " << original_size_ << " to " << matrix_size_ << std::endl;
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
    std::cout << "[DEBUG] Trimmed output matrix to original size " << original_size_ << std::endl;
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
  std::cout << "[DEBUG] Padded matrix from " << original_size << "x" << original_size << " to " << new_size << "x"
            << new_size << std::endl;
  return padded_matrix;
}

std::vector<double> StrassenAll::TrimMatrixToOriginalSize(const std::vector<double>& matrix, int original_size,
                                                          int padded_size) {
  std::vector<double> trimmed_matrix(original_size * original_size);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * padded_size, matrix.begin() + i * padded_size + original_size,
                      trimmed_matrix.begin() + i * original_size);
  }
  std::cout << "[DEBUG] Trimmed matrix from " << padded_size << "x" << padded_size << " to " << original_size << "x"
            << original_size << std::endl;
  return trimmed_matrix;
}

std::vector<double> StrassenAll::StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b, int size,
                                                  int num_threads) {
  static boost::mpi::environment* env = nullptr;
  if (!boost::mpi::environment::initialized()) {
    static int argc = 0;
    static char** argv = nullptr;
    std::cout << "[DEBUG] Initializing MPI environment" << std::endl;
    env = new boost::mpi::environment(argc, argv);
  } else {
    std::cout << "[DEBUG] MPI environment already initialized" << std::endl;
  }

  boost::mpi::communicator world;
  int rank = world.rank();
  int world_size = world.size();
  std::cout << "[DEBUG] Rank: " << rank << ", World size: " << world_size << ", Matrix size: " << size << std::endl;

  // Базовый случай
  if (size <= 32 || world_size <= 1) {
    std::cout << "[DEBUG] Using StandardMultiply (base case, size=" << size << " or world_size=" << world_size << ")"
              << std::endl;
    return StandardMultiply(a, b, size);
  }

  // Разделение матриц
  int half_size = size / 2;
  int half_size_squared = half_size * half_size;
  std::cout << "[DEBUG] Splitting matrices, half_size=" << half_size << std::endl;

  std::vector<double> a11(half_size_squared);
  std::vector<double> a12(half_size_squared);
  std::vector<double> a21(half_size_squared);
  std::vector<double> a22(half_size_squared);
  std::vector<double> b11(half_size_squared);
  std::vector<double> b12(half_size_squared);
  std::vector<double> b21(half_size_squared);
  std::vector<double> b22(half_size_squared);

  std::cout << "[DEBUG] Splitting matrix A" << std::endl;
  SplitMatrix(a, a11, 0, 0, size);
  SplitMatrix(a, a12, 0, half_size, size);
  SplitMatrix(a, a21, half_size, 0, size);
  SplitMatrix(a, a22, half_size, half_size, size);

  std::cout << "[DEBUG] Splitting matrix B" << std::endl;
  SplitMatrix(b, b11, 0, 0, size);
  SplitMatrix(b, b12, 0, half_size, size);
  SplitMatrix(b, b21, half_size, 0, size);
  SplitMatrix(b, b22, half_size, half_size, size);

  std::vector<double> p1, p2, p3, p4, p5, p6, p7;

  // Распределение задач по процессам (1–4)
  int max_processes = std::min(world_size, 4);
  std::cout << "[DEBUG] Max processes for tasks: " << max_processes << std::endl;

  if (rank < max_processes) {
    std::cout << "[DEBUG] Rank " << rank << " assigned task p" << (rank + 1) << std::endl;
    switch (rank) {
      case 0:
        std::cout << "[DEBUG] Rank 0 computing p1" << std::endl;
        p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size,
                              num_threads);
        std::cout << "[DEBUG] Rank 0 finished p1, size=" << p1.size() << std::endl;
        if (!p1.empty()) std::cout << "[DEBUG] p1[0]=" << p1[0] << std::endl;
        break;
      case 1:
        std::cout << "[DEBUG] Rank 1 computing p2" << std::endl;
        p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads);
        std::cout << "[DEBUG] Rank 1 finished p2, size=" << p2.size() << std::endl;
        if (!p2.empty()) std::cout << "[DEBUG] p2[0]=" << p2[0] << std::endl;
        break;
      case 2:
        std::cout << "[DEBUG] Rank 2 computing p3" << std::endl;
        p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads);
        std::cout << "[DEBUG] Rank 2 finished p3, size=" << p3.size() << std::endl;
        if (!p3.empty()) std::cout << "[DEBUG] p3[0]=" << p3[0] << std::endl;
        break;
      case 3:
        std::cout << "[DEBUG] Rank 3 computing p4" << std::endl;
        p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads);
        std::cout << "[DEBUG] Rank 3 finished p4, size=" << p4.size() << std::endl;
        if (!p4.empty()) std::cout << "[DEBUG] p4[0]=" << p4[0] << std::endl;
        break;
    }
  }

  // Синхронизация для p1–p4
  std::cout << "[DEBUG] Rank " << rank << " broadcasting p1-p4" << std::endl;
  if (max_processes > 0) boost::mpi::broadcast(world, p1, 0);
  if (max_processes > 1) boost::mpi::broadcast(world, p2, 1);
  if (max_processes > 2) boost::mpi::broadcast(world, p3, 2);
  if (max_processes > 3) boost::mpi::broadcast(world, p4, 3);
  std::cout << "[DEBUG] Rank " << rank << " received broadcasts for p1-p4" << std::endl;

  // Оставшиеся задачи (p5, p6, p7) выполняются в потоках на rank 0
  std::vector<std::function<void()>> remaining_tasks;
  if (rank == 0) {
    std::cout << "[DEBUG] Rank 0 assigning remaining tasks (p5, p6, p7)" << std::endl;
    remaining_tasks.emplace_back([&]() {
      std::cout << "[DEBUG] Thread computing p5" << std::endl;
      p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads);
      std::cout << "[DEBUG] Thread finished p5, size=" << p5.size() << std::endl;
      if (!p5.empty()) std::cout << "[DEBUG] p5[0]=" << p5[0] << std::endl;
    });
    remaining_tasks.emplace_back([&]() {
      std::cout << "[DEBUG] Thread computing p6" << std::endl;
      p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                            num_threads);
      std::cout << "[DEBUG] Thread finished p6, size=" << p6.size() << std::endl;
      if (!p6.empty()) std::cout << "[DEBUG] p6[0]=" << p6[0] << std::endl;
    });
    remaining_tasks.emplace_back([&]() {
      std::cout << "[DEBUG] Thread computing p7" << std::endl;
      p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                            num_threads);
      std::cout << "[DEBUG] Thread finished p7, size=" << p7.size() << std::endl;
      if (!p7.empty()) std::cout << "[DEBUG] p7[0]=" << p7[0] << std::endl;
    });
  }

  // Запуск потоков
  if (!remaining_tasks.empty()) {
    std::cout << "[DEBUG] Rank " << rank << " launching " << remaining_tasks.size() << " threads for remaining tasks"
              << std::endl;
    std::vector<std::thread> threads;
    threads.reserve(std::min(num_threads, static_cast<int>(remaining_tasks.size())));
    size_t task_index = 0;

    for (int i = 0; i < std::min(num_threads, static_cast<int>(remaining_tasks.size())); ++i) {
      if (task_index < remaining_tasks.size()) {
        std::cout << "[DEBUG] Rank " << rank << " starting thread for task " << task_index << std::endl;
        threads.emplace_back(remaining_tasks[task_index]);
        ++task_index;
      }
    }

    while (task_index < remaining_tasks.size()) {
      std::cout << "[DEBUG] Rank " << rank << " executing task " << task_index << " in main thread" << std::endl;
      remaining_tasks[task_index]();
      ++task_index;
    }

    for (auto& thread : threads) {
      if (thread.joinable()) {
        std::cout << "[DEBUG] Rank " << rank << " joining thread" << std::endl;
        thread.join();
      }
    }
    std::cout << "[DEBUG] Rank " << rank << " finished all threads" << std::endl;
  }

  // Барьер перед p5–p7
  std::cout << "[DEBUG] Rank " << rank << " reaching barrier before p5-p7 broadcast" << std::endl;
  world.barrier();

  // Синхронизация для p5–p7
  if (rank == 0) {
    std::cout << "[DEBUG] Rank 0 broadcasting p5, p6, p7" << std::endl;
  }
  boost::mpi::broadcast(world, p5, 0);
  boost::mpi::broadcast(world, p6, 0);
  boost::mpi::broadcast(world, p7, 0);
  std::cout << "[DEBUG] Rank " << rank << " received p5, p6, p7" << std::endl;

  // Проверка корректности p1–p7
  if (p1.empty() || p2.empty() || p3.empty() || p4.empty() || p5.empty() || p6.empty() || p7.empty()) {
    std::cout << "[ERROR] Rank " << rank << " detected empty p1-p7 vectors" << std::endl;
    return std::vector<double>(size * size, 0.0);  // Возвращаем нулевую матрицу в случае ошибки
  }

  // Формирование результата
  std::cout << "[DEBUG] Rank " << rank << " combining results" << std::endl;
  std::vector<double> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  std::vector<double> c12 = AddMatrices(p3, p5, half_size);
  std::vector<double> c21 = AddMatrices(p2, p4, half_size);
  std::vector<double> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

  std::cout << "[DEBUG] Rank " << rank << " c11[0]=" << (c11.empty() ? 0 : c11[0])
            << ", c12[0]=" << (c12.empty() ? 0 : c12[0]) << ", c21[0]=" << (c21.empty() ? 0 : c21[0])
            << ", c22[0]=" << (c22.empty() ? 0 : c22[0]) << std::endl;

  std::vector<double> result(size * size);
  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);
  std::cout << "[DEBUG] Rank " << rank << " result matrix merged, size=" << result.size() << std::endl;
  if (!result.empty()) std::cout << "[DEBUG] result[0]=" << result[0] << std::endl;

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