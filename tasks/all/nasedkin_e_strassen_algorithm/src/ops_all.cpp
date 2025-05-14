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
  if (world_.rank() == 0) {
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
  }

  // Broadcast matrix size and original size to all processes
  boost::mpi::broadcast(world_, matrix_size_, 0);
  boost::mpi::broadcast(world_, original_size_, 0);

  // Reserve memory for matrices on all processes
  if (world_.rank() != 0) {
    input_matrix_a_.resize(matrix_size_ * matrix_size_);
    input_matrix_b_.resize(matrix_size_ * matrix_size_);
    output_matrix_.resize(matrix_size_ * matrix_size_, 0.0);
  }

  // Broadcast matrices to all processes
  boost::mpi::broadcast(world_, input_matrix_a_, 0);
  boost::mpi::broadcast(world_, input_matrix_b_, 0);

  return true;
}

bool StrassenAll::ValidationImpl() {
  // Validation only needs to be done by the root process
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

  // Non-root processes just return true as root will determine validation
  return true;
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

  // Распределение задач по MPI процессам
  int world_size = world_.size();
  int rank = world_.rank();

  // Подготовим все необходимые вычисления заранее
  std::vector<std::function<std::vector<double>()>> computations;
  computations.reserve(7);

  computations.emplace_back([&]() {
    return StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, num_threads);
  });
  computations.emplace_back(
      [&]() { return StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads); });
  computations.emplace_back(
      [&]() { return StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads); });
  computations.emplace_back(
      [&]() { return StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads); });
  computations.emplace_back(
      [&]() { return StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads); });
  computations.emplace_back([&]() {
    return StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                            num_threads);
  });
  computations.emplace_back([&]() {
    return StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                            num_threads);
  });

  // Вектор для хранения результатов вычислений
  std::vector<std::vector<double>> results(7);

  // Распределим вычисления между процессами и потоками
  // Первый уровень распараллеливания: MPI процессы
  if (world_size > 1) {
    // Распределение задач по MPI процессам
    int tasks_per_process = 7 / world_size + (7 % world__size > 0 ? 1 : 0);
    int start_task = rank * tasks_per_process;
    int end_task = std::min(static_cast<int>(computations.size()), (rank + 1) * tasks_per_process);

    // Выполнение задач, назначенных текущему процессу
    for (int i = start_task; i < end_task; ++i) {
      results[i] = computations[i]();
    }

    // Сбор результатов от всех процессов
    for (int i = 0; i < 7; ++i) {
      int source_rank = i / tasks_per_process;
      if (source_rank >= world_size) source_rank = world__size - 1;

      if (rank == source_rank && i >= start_task && i < end_task) {
        // Отправка результата корневому процессу
        if (rank != 0) {
          boost::mpi::send(world_, 0, i, results[i]);
        }
      } else if (rank == 0) {
        // Корневой процесс принимает результат
        if (source_rank != 0) {
          boost::mpi::recv(world_, source_rank, i, results[i]);
        }
      }
    }
  } else {
    // Если только один процесс, используем только многопоточность
    std::vector<std::thread> threads;
    threads.reserve(std::min(num_threads, static_cast<int>(computations.size())));

    // Распределение задач по потокам
    size_t task_index = 0;
    for (int i = 0; i < std::min(num_threads, static_cast<int>(computations.size())); ++i) {
      if (task_index < computations.size()) {
        threads.emplace_back(
            [&results, &computations, task_index]() { results[task_index] = computations[task_index](); });
        ++task_index;
      }
    }

    // Выполнение оставшихся задач последовательно
    while (task_index < computations.size()) {
      results[task_index] = computations[task_index]();
      ++task_index;
    }

    // Ожидание завершения всех потоков
    for (auto& thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  // Синхронизация между всеми процессами
  boost::mpi::barrier(world_);

  // Формирование конечного результата только в корневом процессе
  std::vector<double> result(size * size);

  if (rank == 0) {
    // Распаковка результатов
    p1 = results[0];
    p2 = results[1];
    p3 = results[2];
    p4 = results[3];
    p5 = results[4];
    p6 = results[5];
    p7 = results[6];

    std::vector<double> c11 =
        AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
    std::vector<double> c12 = AddMatrices(p3, p5, half_size);
    std::vector<double> c21 = AddMatrices(p2, p4, half_size);
    std::vector<double> c22 =
        AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

    MergeMatrix(result, c11, 0, 0, size);
    MergeMatrix(result, c12, 0, half_size, size);
    MergeMatrix(result, c21, half_size, 0, size);
    MergeMatrix(result, c22, half_size, half_size, size);
  }

  // Рассылаем результат всем процессам
  boost::mpi::broadcast(world_, result, 0);

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