#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace nasedkin_e_strassen_algorithm_all {

bool StrassenAll::PreProcessingImpl() {
  std::cout << "Starting PreProcessingImpl" << std::endl;
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* in_ptr_b = reinterpret_cast<double*>(task_data->inputs[1]);

  matrix_size_ = static_cast<int>(std::sqrt(input_size));
  input_matrix_a_.resize(matrix_size_ * matrix_size_);
  input_matrix_b_.resize(matrix_size_ * matrix_size_);

  std::cout << "Matrix size: " << matrix_size_ << std::endl;
  std::ranges::copy(in_ptr_a, in_ptr_a + input_size, input_matrix_a_.begin());
  std::ranges::copy(in_ptr_b, in_ptr_b + input_size, input_matrix_b_.begin());

  if ((matrix_size_ & (matrix_size_ - 1)) != 0) {
    std::cout << "Padding matrix to power of two" << std::endl;
    original_size_ = matrix_size_;
    input_matrix_a_ = PadMatrixToPowerOfTwo(input_matrix_a_, matrix_size_);
    input_matrix_b_ = PadMatrixToPowerOfTwo(input_matrix_b_, matrix_size_);
    matrix_size_ = static_cast<int>(std::sqrt(input_matrix_a_.size()));
    std::cout << "New matrix size after padding: " << matrix_size_ << std::endl;
  } else {
    original_size_ = matrix_size_;
  }

  output_matrix_.resize(matrix_size_ * matrix_size_, 0.0);
  std::cout << "PreProcessingImpl completed" << std::endl;
  return true;
}

bool StrassenAll::ValidationImpl() {
  std::cout << "Starting ValidationImpl" << std::endl;
  unsigned int input_size_a = task_data->inputs_count[0];
  unsigned int input_size_b = task_data->inputs_count[1];
  unsigned int output_size = task_data->outputs_count[0];

  if (input_size_a == 0 || input_size_b == 0 || output_size == 0) {
    std::cout << "Validation failed: zero input/output size" << std::endl;
    return false;
  }

  int size_a = static_cast<int>(std::sqrt(input_size_a));
  int size_b = static_cast<int>(std::sqrt(input_size_b));
  int size_output = static_cast<int>(std::sqrt(output_size));

  bool valid = (size_a == size_b) && (size_a == size_output);
  std::cout << "Validation result: " << (valid ? "valid" : "invalid") << std::endl;
  return valid;
}

bool StrassenAll::RunImpl() {
  std::cout << "Starting RunImpl" << std::endl;
  int num_threads = std::min(16, ppc::util::GetPPCNumThreads());
  std::cout << "Number of threads: " << num_threads << std::endl;
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_, num_threads);
  std::cout << "RunImpl completed" << std::endl;
  return true;
}

bool StrassenAll::PostProcessingImpl() {
  std::cout << "Starting PostProcessingImpl" << std::endl;
  if (original_size_ != matrix_size_) {
    std::cout << "Trimming matrix to original size: " << original_size_ << std::endl;
    output_matrix_ = TrimMatrixToOriginalSize(output_matrix_, original_size_, matrix_size_);
  }

  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(output_matrix_, out_ptr);
  std::cout << "PostProcessingImpl completed" << std::endl;
  return true;
}

std::vector<double> StrassenAll::AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::cout << "Adding matrices of size: " << size << std::endl;
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::plus<>());
  return result;
}

std::vector<double> StrassenAll::SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b,
                                                  int size) {
  std::cout << "Subtracting matrices of size: " << size << std::endl;
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::minus<>());
  return result;
}

std::vector<double> StandardMultiply(const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::cout << "Performing standard matrix multiplication of size: " << size << std::endl;
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
  std::cout << "Padding matrix from size " << original_size << " to power of two" << std::endl;
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
  std::cout << "Trimming matrix from size " << padded_size << " to " << original_size << std::endl;
  std::vector<double> trimmed_matrix(original_size * original_size);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * padded_size, matrix.begin() + i * padded_size + original_size,
                      trimmed_matrix.begin() + i * original_size);
  }
  return trimmed_matrix;
}

std::vector<double> StrassenAll::StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b, int size,
                                                  int num_threads) {
  std::cout << "Starting StrassenMultiply for matrix size: " << size << " with " << num_threads << " threads"
            << std::endl;

  // Инициализация MPI
  boost::mpi::environment* env = nullptr;
  if (!boost::mpi::environment::initialized()) {
    std::cout << "Initializing MPI environment" << std::endl;
    static int argc = 0;
    static char** argv = nullptr;
    env = new boost::mpi::environment(argc, argv);
  }

  boost::mpi::communicator world;
  int rank = world.rank();
  int world_size = world.size();
  std::cout << "MPI Rank: " << rank << ", World size: " << world_size << std::endl;

  // Базовый случай
  if (size <= 32 || world_size <= 1) {
    std::cout << "Using standard multiplication for small matrix or single process" << std::endl;
    auto result = StandardMultiply(a, b, size);
    if (env != nullptr) {
      std::cout << "Deleting MPI environment" << std::endl;
      delete env;
    }
    return result;
  }

  int half_size = size / 2;
  int half_size_squared = half_size * half_size;
  std::cout << "Half size: " << half_size << std::endl;

  // Разделение матриц
  std::vector<double> a11(half_size_squared);
  std::vector<double> a12(half_size_squared);
  std::vector<double> a21(half_size_squared);
  std::vector<double> a22(half_size_squared);
  std::vector<double> b11(half_size_squared);
  std::vector<double> b12(half_size_squared);
  std::vector<double> b21(half_size_squared);
  std::vector<double> b22(half_size_squared);

  std::cout << "Splitting matrices" << std::endl;
  SplitMatrix(a, a11, 0, 0, size);
  SplitMatrix(a, a12, 0, half_size, size);
  SplitMatrix(a, a21, half_size, 0, size);
  SplitMatrix(a, a22, half_size, half_size, size);
  SplitMatrix(b, b11, 0, 0, size);
  SplitMatrix(b, b12, 0, half_size, size);
  SplitMatrix(b, b21, half_size, 0, size);
  SplitMatrix(b, b22, half_size, half_size, size);

  std::vector<double> p1, p2, p3, p4, p5, p6, p7;

  // Определяем задачи для каждого процесса
  std::vector<std::function<void()>> tasks;
  tasks.emplace_back([&]() {
    std::cout << "Rank " << rank << ": Computing p1" << std::endl;
    p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, num_threads);
  });
  tasks.emplace_back([&]() {
    std::cout << "Rank " << rank << ": Computing p2" << std::endl;
    p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads);
  });
  tasks.emplace_back([&]() {
    std::cout << "Rank " << rank << ": Computing p3" << std::endl;
    p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads);
  });
  tasks.emplace_back([&]() {
    std::cout << "Rank " << rank << ": Computing p4" << std::endl;
    p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads);
  });
  tasks.emplace_back([&]() {
    std::cout << "Rank " << rank << ": Computing p5" << std::endl;
    p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads);
  });
  tasks.emplace_back([&]() {
    std::cout << "Rank " << rank << ": Computing p6" << std::endl;
    p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                          num_threads);
  });
  tasks.emplace_back([&]() {
    std::cout << "Rank " << rank << ": Computing p7" << std::endl;
    p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                          num_threads);
  });

  // Распределяем задачи по процессам
  std::vector<std::function<void()>> local_tasks;
  int total_tasks = 7;
  int tasks_per_process = total_tasks / world_size;
  int extra_tasks = total_tasks % world_size;
  int start_task = rank * tasks_per_process + std::min(rank, extra_tasks);
  int end_task = start_task + tasks_per_process + (rank < extra_tasks ? 1 : 0);

  std::cout << "Rank " << rank << ": Assigned tasks from " << start_task << " to " << end_task - 1 << std::endl;

  // Выполняем назначенные задачи с использованием многопоточности
  std::vector<std::thread> threads;
  threads.reserve(std::min(num_threads, end_task - start_task));
  for (int i = start_task; i < end_task; ++i) {
    if (threads.size() < static_cast<size_t>(num_threads)) {
      threads.emplace_back(tasks[i]);
    } else {
      tasks[i]();
    }
  }
  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  // Собираем результаты через MPI broadcast
  std::cout << "Rank " << rank << ": Starting MPI broadcasts" << std::endl;
  for (int i = 0; i < total_tasks; ++i) {
    int root = i % world_size;
    if (i >= start_task && i < end_task) {
      switch (i) {
        case 0:
          boost::mpi::broadcast(world, p1, root);
          break;
        case 1:
          boost::mpi::broadcast(world, p2, root);
          break;
        case 2:
          boost::mpi::broadcast(world, p3, root);
          break;
        case 3:
          boost::mpi::broadcast(world, p4, root);
          break;
        case 4:
          boost::mpi::broadcast(world, p5, root);
          break;
        case 5:
          boost::mpi::broadcast(world, p6, root);
          break;
        case 6:
          boost::mpi::broadcast(world, p7, root);
          break;
      }
    } else {
      switch (i) {
        case 0:
          boost::mpi::broadcast(world, p1, root);
          break;
        case 1:
          boost::mpi::broadcast(world, p2, root);
          break;
        case 2:
          boost::mpi::broadcast(world, p3, root);
          break;
        case 3:
          boost::mpi::broadcast(world, p4, root);
          break;
        case 4:
          boost::mpi::broadcast(world, p5, root);
          break;
        case 5:
          boost::mpi::broadcast(world, p6, root);
          break;
        case 6:
          boost::mpi::broadcast(world, p7, root);
          break;
      }
    }
  }
  std::cout << "Rank " << rank << ":  MPI broadcasts completed" << std::endl;
  std::cout << "Rank " << rank << ": Combining results" << std::endl;
  std::vector<double> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  std::vector<double> c12 = AddMatrices(p3, p5, half_size);
  std::vector<double> c21 = AddMatrices(p2, p4, half_size);
  std::vector<double> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

  std::vector<double> result(size * size);
  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);

  if (env != nullptr) {
    std::cout << "Deleting MPI environment" << std::endl;
    delete env;
  }

  std::cout << "StrassenMultiply completed" << std::endl;
  return result;
}

void StrassenAll::SplitMatrix(const std::vector<double>& parent, std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  std::cout << "Splitting matrix at (" << row_start << ", " << col_start << ")" << std::endl;
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(parent.begin() + (row_start + i) * parent_size + col_start,
                      parent.begin() + (row_start + i) * parent_size + col_start + child_size,
                      child.begin() + i * child_size);
  }
}

void StrassenAll::MergeMatrix(std::vector<double>& parent, const std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  std::cout << "Merging matrix at (" << row_start << ", " << col_start << ")" << std::endl;
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(child.begin() + i * child_size, child.begin() + (i + 1) * child_size,
                      parent.begin() + (row_start + i) * parent_size + col_start);
  }
}

}  // namespace nasedkin_e_strassen_algorithm_all