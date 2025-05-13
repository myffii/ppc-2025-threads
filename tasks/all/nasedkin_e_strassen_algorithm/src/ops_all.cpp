#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
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

  std::cout << "[DEBUG] PreProcessing: Input size = " << input_size << std::endl;

  matrix_size_ = static_cast<int>(std::sqrt(input_size));
  input_matrix_a_.resize(matrix_size_ * matrix_size_);
  input_matrix_b_.resize(matrix_size_ * matrix_size_);

  std::ranges::copy(in_ptr_a, in_ptr_a + input_size, input_matrix_a_.begin());
  std::ranges::copy(in_ptr_b, in_ptr_b + input_size, input_matrix_b_.begin());

  if ((matrix_size_ & (matrix_size_ - 1)) != 0) {
    std::cout << "[DEBUG] PreProcessing: Padding matrix from size " << matrix_size_ << std::endl;
    original_size_ = matrix_size_;
    input_matrix_a_ = PadMatrixToPowerOfTwo(input_matrix_a_, matrix_size_);
    input_matrix_b_ = PadMatrixToPowerOfTwo(input_matrix_b_, matrix_size_);
    matrix_size_ = static_cast<int>(std::sqrt(input_matrix_a_.size()));
    std::cout << "[DEBUG] PreProcessing: Padded to size " << matrix_size_ << std::endl;
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

  std::cout << "[DEBUG] Validation: input_size_a = " << input_size_a << ", input_size_b = " << input_size_b
            << ", output_size = " << output_size << std::endl;

  if (input_size_a == 0 || input_size_b == 0 || output_size == 0) {
    std::cout << "[DEBUG] Validation: Failed due to zero size" << std::endl;
    return false;
  }

  int size_a = static_cast<int>(std::sqrt(input_size_a));
  int size_b = static_cast<int>(std::sqrt(input_size_b));
  int size_output = static_cast<int>(std::sqrt(output_size));

  bool valid = (size_a == size_b) && (size_a == size_output);
  std::cout << "[DEBUG] Validation: Result = " << (valid ? "true" : "false") << std::endl;
  return valid;
}

bool StrassenAll::RunImpl() {
  boost::mpi::communicator world;
  std::cout << "[DEBUG] Run: MPI Rank = " << world.rank() << ", Size = " << world.size() << std::endl;
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_, world);
  return true;
}

bool StrassenAll::PostProcessingImpl() {
  if (original_size_ != matrix_size_) {
    std::cout << "[DEBUG] PostProcessing: Trimming matrix from size " << matrix_size_ << " to " << original_size_
              << std::endl;
    output_matrix_ = TrimMatrixToOriginalSize(output_matrix_, original_size_, matrix_size_);
  }

  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(output_matrix_, out_ptr);
  std::cout << "[DEBUG] PostProcessing: Output copied" << std::endl;
  return true;
}

std::vector<double> StrassenAll::AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::plus<>());
  std::cout << "[DEBUG] AddMatrices: Size = " << size << std::endl;
  return result;
}

std::vector<double> StrassenAll::SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b,
                                                  int size) {
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::minus<>());
  std::cout << "[DEBUG] SubtractMatrices: Size = " << size << std::endl;
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
  std::cout << "[DEBUG] StandardMultiply: Size = " << size << std::endl;
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
  std::cout << "[DEBUG] PadMatrixToPowerOfTwo: From " << original_size << " to " << new_size << std::endl;
  return padded_matrix;
}

std::vector<double> StrassenAll::TrimMatrixToOriginalSize(const std::vector<double>& matrix, int original_size,
                                                          int padded_size) {
  std::vector<double> trimmed_matrix(original_size * original_size);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * padded_size, matrix.begin() + i * padded_size + original_size,
                      trimmed_matrix.begin() + i * original_size);
  }
  std::cout << "[DEBUG] TrimMatrixToOriginalSize: From " << padded_size << " to " << original_size << std::endl;
  return trimmed_matrix;
}

std::vector<double> StrassenAll::StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b, int size,
                                                  boost::mpi::communicator& world) {
  if (size <= 32) {
    std::cout << "[DEBUG] StrassenMultiply: Using StandardMultiply for size = " << size << std::endl;
    return StandardMultiply(a, b, size);
  }

  std::cout << "[DEBUG] StrassenMultiply: Rank = " << world.rank() << ", Size = " << size << std::endl;

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

  std::cout << "[DEBUG] StrassenMultiply: Rank = " << world.rank() << ", Matrices splitted" << std::endl;

  std::vector<double> p1(half_size_squared);
  std::vector<double> p2(half_size_squared);
  std::vector<double> p3(half_size_squared);
  std::vector<double> p4(half_size_squared);
  std::vector<double> p5(half_size_squared);
  std::vector<double> p6(half_size_squared);
  std::vector<double> p7(half_size_squared);

  // Определяем все задачи для вычисления p1-p7
  std::vector<std::function<void()>> tasks;
  tasks.push_back([&]() {
    p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, world);
  });
  tasks.push_back([&]() { p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, world); });
  tasks.push_back([&]() { p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, world); });
  tasks.push_back([&]() { p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, world); });
  tasks.push_back([&]() { p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, world); });
  tasks.push_back([&]() {
    p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size, world);
  });
  tasks.push_back([&]() {
    p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size, world);
  });

  std::cout << "[DEBUG] StrassenMultiply: Rank = " << world.rank() << ", Tasks assigned: " << tasks.size() << std::endl;

  // Распределяем задачи между MPI-процессами
  std::vector<std::function<void()>> local_tasks;
  int tasks_per_process = (tasks.size() + world.size() - 1) / world.size();
  int start_idx = world.rank() * tasks_per_process;
  int end_idx = std::min(static_cast<int>(tasks.size()), start_idx + tasks_per_process);
  for (int i = start_idx; i < end_idx; ++i) {
    local_tasks.push_back(tasks[i]);
  }

  std::cout << "[DEBUG] StrassenMultiply: Rank = " << world.rank() << ", Local tasks: " << local_tasks.size()
            << std::endl;

  // Параллелим локальные задачи с помощью потоков
  int num_threads = ppc::util::GetPPCNumThreads();
  std::cout << "[DEBUG] StrassenMultiply: Rank = " << world.rank() << ", Using " << num_threads << " threads"
            << std::endl;

  std::vector<std::thread> threads;
  for (size_t i = 0; i < local_tasks.size(); ++i) {
    if (i < static_cast<size_t>(num_threads)) {
      threads.emplace_back(local_tasks[i]);
    } else {
      local_tasks[i]();  // Выполняем последовательно, если потоков не хватает
    }
  }

  for (auto& t : threads) {
    t.join();
  }

  std::cout << "[DEBUG] StrassenMultiply: Rank = " << world.rank() << ", Threads joined" << std::endl;

  // Собираем результаты через MPI
  if (world.rank() == 0) {
    // Уже вычисленные p1-p7 для rank 0
    if (world.size() > 1) {
      for (int rank = 1; rank < world.size(); ++rank) {
        int rank_start_idx = rank * tasks_per_process;
        int rank_end_idx = std::min(static_cast<int>(tasks.size()), rank_start_idx + tasks_per_process);
        for (int i = rank_start_idx; i < rank_end_idx; ++i) {
          if (i == 0) continue;  // p1 уже у rank 0
          std::vector<double> temp(half_size_squared);
          world.recv(rank, i, temp.data(), half_size_squared);
          std::cout << "[DEBUG] StrassenMultiply: Rank 0 received p" << i + 1 << " from rank " << rank << std::endl;
          if (i == 1)
            p2 = temp;
          else if (i == 2)
            p3 = temp;
          else if (i == 3)
            p4 = temp;
          else if (i == 4)
            p5 = temp;
          else if (i == 5)
            p6 = temp;
          else if (i == 6)
            p7 = temp;
        }
      }
    }
  } else {
    // Отправляем вычисленные p_i на rank 0
    for (size_t i = 0; i < local_tasks.size(); ++i) {
      int global_idx = start_idx + i;
      if (global_idx == 0) continue;  // p1 не отправляем, он у rank 0
      std::vector<double> temp;
      if (global_idx == 1)
        temp = p2;
      else if (global_idx == 2)
        temp = p3;
      else if (global_idx == 3)
        temp = p4;
      else if (global_idx == 4)
        temp = p5;
      else if (global_idx == 5)
        temp = p6;
      else if (global_idx == 6)
        temp = p7;
      if (!temp.empty()) {
        world.send(0, global_idx, temp.data(), half_size_squared);
        std::cout << "[DEBUG] StrassenMultiply: Rank " << world.rank() << " sent p" << global_idx + 1 << std::endl;
      }
    }
  }

  world.barrier();
  std::cout << "[DEBUG] StrassenMultiply: Rank = " << world.rank() << ", MPI Barrier passed" << std::endl;

  // Рассылаем p1-p7 всем процессам
  if (world.rank() == 0) {
    boost::mpi::broadcast(world, p1.data(), half_size_squared, 0);
    boost::mpi::broadcast(world, p2.data(), half_size_squared, 0);
    boost::mpi::broadcast(world, p3.data(), half_size_squared, 0);
    boost::mpi::broadcast(world, p4.data(), half_size_squared, 0);
    boost::mpi::broadcast(world, p5.data(), half_size_squared, 0);
    boost::mpi::broadcast(world, p6.data(), half_size_squared, 0);
    boost::mpi::broadcast(world, p7.data(), half_size_squared, 0);
  } else {
    p1.resize(half_size_squared);
    p2.resize(half_size_squared);
    p3.resize(half_size_squared);
    p4.resize(half_size_squared);
    p5.resize(half_size_squared);
    p6.resize(half_size_squared);
    p7.resize(half_size_squared);
    boost::mpi::broadcast(world, p1.data(), half_size_squared, 0);
    boost::mpi::broadcast(world, p2.data(), half_size_squared, 0);
    boost::mpi::broadcast(world, p3.data(), half_size_squared, 0);
    boost::mpi::broadcast(world, p4.data(), half_size_squared, 0);
    boost::mpi::broadcast(world, p5.data(), half_size_squared, 0);
    boost::mpi::broadcast(world, p6.data(), half_size_squared, 0);
    boost::mpi::broadcast(world, p7.data(), half_size_squared, 0);
  }

  std::cout << "[DEBUG] StrassenMultiply: Rank = " << world.rank() << ", p1-p7 broadcasted" << std::endl;

  std::vector<double> result(size * size);
  std::vector<double> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  std::vector<double> c12 = AddMatrices(p3, p5, half_size);
  std::vector<double> c21 = AddMatrices(p2, p4, half_size);
  std::vector<double> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);

  std::cout << "[DEBUG] StrassenMultiply: Rank = " << world.rank() << ", Result matrix computed" << std::endl;

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
  std::cout << "[DEBUG] SplitMatrix: Row_start = " << row_start << ", Col_start = " << col_start << std::endl;
}

void StrassenAll::MergeMatrix(std::vector<double>& parent, const std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(child.begin() + i * child_size, child.begin() + (i + 1) * child_size,
                      parent.begin() + (row_start + i) * parent_size + col_start);
  }
  std::cout << "[DEBUG] MergeMatrix: Row_start = " << row_start << ", Col_start = " << col_start << std::endl;
}

}  // namespace nasedkin_e_strassen_algorithm_all