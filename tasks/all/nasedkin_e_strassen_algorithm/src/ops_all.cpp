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
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_, world);
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
                                                  boost::mpi::communicator& world) {
  if (size <= 32) {
    if (world.rank() == 0) {
      std::cout << "[DEBUG] Size <= 32, using StandardMultiply on rank " << world.rank() << std::endl;
    }
    return StandardMultiply(a, b, size);
  }

  int half_size = size / 2;
  int half_size_squared = half_size * half_size;

  std::cout << "[DEBUG] Rank " << world.rank() << ": Splitting matrices, half_size = " << half_size << std::endl;

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

  std::vector<double> p1(half_size_squared);
  std::vector<double> p2(half_size_squared);
  std::vector<double> p3(half_size_squared);
  std::vector<double> p4(half_size_squared);
  std::vector<double> p5(half_size_squared);
  std::vector<double> p6(half_size_squared);
  std::vector<double> p7(half_size_squared);

  // Распределение задач
  const int total_tasks = 7;
  int world_size = world.size();
  if (world_size > total_tasks) world_size = total_tasks;  // Ограничиваем до 7 процессов
  int tasks_per_process = 1;                               // Каждый процесс получает минимум 1 задачу
  int remaining_tasks = total_tasks - world_size;          // Остаток задач для потоков

  std::cout << "[DEBUG] Rank " << world.rank() << ": world_size = " << world_size
            << ", remaining_tasks for threads = " << remaining_tasks << std::endl;

  int num_threads = ppc::util::GetPPCNumThreads();
  std::cout << "[DEBUG] Rank " << world.rank() << ": Number of threads = " << num_threads << std::endl;

  std::vector<std::thread> threads;
  std::vector<std::vector<double>> thread_results(remaining_tasks);

  // Выполнение задачи текущего процесса
  if (world.rank() < total_tasks) {
    std::cout << "[DEBUG] Rank " << world.rank() << ": Executing task p" << world.rank() + 1 << std::endl;
    switch (world.rank()) {
      case 0:
        p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, world);
        break;
      case 1:
        p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, world);
        break;
      case 2:
        p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, world);
        break;
      case 3:
        p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, world);
        break;
      case 4:
        p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, world);
        break;
      case 5:
        p6 =
            StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size, world);
        break;
      case 6:
        p7 =
            StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size, world);
        break;
    }
  }

  // Распределение оставшихся задач по потокам на процессе 0
  if (world.rank() == 0 && remaining_tasks > 0) {
    std::cout << "[DEBUG] Rank 0: Distributing " << remaining_tasks << " tasks to threads" << std::endl;
    int thread_task_start = world_size;
    for (int i = 0; i < remaining_tasks; ++i) {
      int task_id = thread_task_start + i;
      threads.emplace_back([&, task_id, i]() {
        std::cout << "[DEBUG] Rank 0, Thread " << std::this_thread::get_id() << ": Executing task p" << task_id + 1
                  << std::endl;
        switch (task_id) {
          case 0:
            thread_results[i] =
                StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, world);
            break;
          case 1:
            thread_results[i] = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, world);
            break;
          case 2:
            thread_results[i] = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, world);
            break;
          case 3:
            thread_results[i] = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, world);
            break;
          case 4:
            thread_results[i] = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, world);
            break;
          case 5:
            thread_results[i] = StrassenMultiply(SubtractMatrices(a21, a11, half_size),
                                                 AddMatrices(b11, b12, half_size), half_size, world);
            break;
          case 6:
            thread_results[i] = StrassenMultiply(SubtractMatrices(a12, a22, half_size),
                                                 AddMatrices(b21, b22, half_size), half_size, world);
            break;
        }
      });
    }
  }

  // Ожидание завершения потоков
  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
      std::cout << "[DEBUG] Rank " << world.rank() << ": Thread joined" << std::endl;
    }
  }

  // Сбор результатов от всех процессов
  if (world.rank() == 0) {
    std::cout << "[DEBUG] Rank 0: Collecting results" << std::endl;
    // Присваиваем результаты из потоков
    for (int i = 0; i < remaining_tasks; ++i) {
      int task_id = world_size + i;
      switch (task_id) {
        case 0:
          p1 = thread_results[i];
          break;
        case 1:
          p2 = thread_results[i];
          break;
        case 2:
          p3 = thread_results[i];
          break;
        case 3:
          p4 = thread_results[i];
          break;
        case 4:
          p5 = thread_results[i];
          break;
        case 5:
          p6 = thread_results[i];
          break;
        case 6:
          p7 = thread_results[i];
          break;
      }
    }
    // Получение результатов от других процессов
    for (int i = 1; i < world_size && i < total_tasks; ++i) {
      std::vector<double> received(half_size_squared);
      world.recv(i, i, received);
      std::cout << "[DEBUG] Rank 0: Received result for p" << i + 1 << " from rank " << i << std::endl;
      switch (i) {
        case 0:
          p1 = received;
          break;
        case 1:
          p2 = received;
          break;
        case 2:
          p3 = received;
          break;
        case 3:
          p4 = received;
          break;
        case 4:

          p5 = received;
          break;
        case 5:
          p6 = received;
          break;
        case 6:
          p7 = received;
          break;
      }
    }
  } else if (world.rank() < total_tasks) {
    std::cout << "[DEBUG] Rank " << world.rank() << ": Sending result for p" << world.rank() + 1 << " to rank 0"
              << std::endl;
    switch (world.rank()) {
      case 0:
        world.send(0, 0, p1);
        break;
      case 1:
        world.send(0, 1, p2);
        break;
      case 2:
        world.send(0, 2, p3);
        break;
      case 3:
        world.send(0, 3, p4);
        break;
      case 4:
        world.send(0, 4, p5);
        break;
      case 5:
        world.send(0, 5, p6);
        break;
      case 6:
        world.send(0, 6, p7);
        break;
    }
  }

  world.barrier();
  std::cout << "[DEBUG] Rank " << world.rank() << ": Passed MPI barrier" << std::endl;

  std::vector<double> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  std::vector<double> c12 = AddMatrices(p3, p5, half_size);
  std::vector<double> c21 = AddMatrices(p2, p4, half_size);
  std::vector<double> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

  std::vector<double> result(size * size);
  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);

  std::cout << "[DEBUG] Rank " << world.rank() << ": StrassenMultiply completed, result size = " << result.size()
            << std::endl;
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