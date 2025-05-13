#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <mutex>
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
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_);
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

std::vector<double> StrassenAll::StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b,
                                                  int size) {
  // Получаем ранг и размерность MPI
  boost::mpi::communicator world;
  int rank = world.rank();
  int num_processes = world.size();

  std::cout << "Process " << rank << " of " << num_processes << " started StrassenMultiply with matrix size " << size
            << std::endl;

  // Для небольших матриц используем стандартное умножение
  if (size <= 32) {
    std::cout << "Process " << rank << ": Using standard multiplication for size " << size << std::endl;
    return StandardMultiply(a, b, size);
  }

  int half_size = size / 2;
  int half_size_squared = half_size * half_size;

  // Разделяем матрицы на подматрицы
  std::vector<double> a11(half_size_squared);
  std::vector<double> a12(half_size_squared);
  std::vector<double> a21(half_size_squared);
  std::vector<double> a22(half_size_squared);
  std::vector<double> b11(half_size_squared);
  std::vector<double> b12(half_size_squared);
  std::vector<double> b21(half_size_squared);
  std::vector<double> b22(half_size_squared);

  std::cout << "Process " << rank << ": Splitting matrices" << std::endl;

  SplitMatrix(a, a11, 0, 0, size);
  SplitMatrix(a, a12, 0, half_size, size);
  SplitMatrix(a, a21, half_size, 0, size);
  SplitMatrix(a, a22, half_size, half_size, size);

  SplitMatrix(b, b11, 0, 0, size);
  SplitMatrix(b, b12, 0, half_size, size);
  SplitMatrix(b, b21, half_size, 0, size);
  SplitMatrix(b, b22, half_size, half_size, size);

  // Подготавливаем промежуточные результаты
  std::vector<double> p1(half_size_squared);
  std::vector<double> p2(half_size_squared);
  std::vector<double> p3(half_size_squared);
  std::vector<double> p4(half_size_squared);
  std::vector<double> p5(half_size_squared);
  std::vector<double> p6(half_size_squared);
  std::vector<double> p7(half_size_squared);

  // Распределяем вычисления p1-p7 между MPI процессами и потоками
  // Определяем количество доступных потоков
  int num_threads = ppc::util::GetPPCNumThreads();
  std::cout << "Process " << rank << ": Using " << num_threads << " threads" << std::endl;

  // Сначала распределяем между MPI процессами
  // Каждый процесс выполняет (7 / num_processes) + остаток операций
  int operations_per_process = 7 / num_processes;
  int remainder = 7 % num_processes;
  int start_op = rank * operations_per_process + (rank < remainder ? rank : remainder);
  int end_op = start_op + operations_per_process + (rank < remainder ? 1 : 0);

  std::cout << "Process " << rank << ": Handling operations from " << start_op << " to " << end_op - 1 << std::endl;

  // Создаем вектор для хранения результатов p1-p7
  std::vector<std::vector<double>> p_results(7);

  // Функция для вычисления отдельных p-выражений
  auto compute_p = [&](int p_idx) {
    std::cout << "Process " << rank << ", Thread handling p" << p_idx + 1 << std::endl;

    switch (p_idx) {
      case 0:  // p1
        return StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size);
      case 1:  // p2
        return StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size);
      case 2:  // p3
        return StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size);
      case 3:  // p4
        return StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size);
      case 4:  // p5
        return StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size);
      case 5:  // p6
        return StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size);
      case 6:  // p7
        return StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size);
      default:
        std::cout << "Process " << rank << ": Invalid p_idx " << p_idx << std::endl;
        return std::vector<double>(half_size_squared, 0.0);
    }
  };

  // Выполняем операции, назначенные данному процессу
  std::vector<std::thread> threads;
  std::mutex cout_mutex;  // Для синхронизации вывода

  // Внутри каждого процесса распределяем вычисления между потоками
  for (int i = start_op; i < end_op; ++i) {
    if (threads.size() < static_cast<size_t>(num_threads - 1) && (end_op - i) > 1) {
      // Запускаем задачу в отдельном потоке, если есть доступные потоки
      threads.emplace_back([i, &p_results, &compute_p, &cout_mutex, rank]() {
        {
          std::lock_guard<std::mutex> lock(cout_mutex);
          std::cout << "Process " << rank << ": Thread started for p" << i + 1 << std::endl;
        }
        p_results[i] = compute_p(i);
        {
          std::lock_guard<std::mutex> lock(cout_mutex);
          std::cout << "Process " << rank << ": Thread completed p" << i + 1 << std::endl;
        }
      });
    } else {
      // Выполняем текущим потоком, если нет доступных потоков
      std::cout << "Process " << rank << ": Main thread handling p" << i + 1 << std::endl;
      p_results[i] = compute_p(i);
      std::cout << "Process " << rank << ": Main thread completed p" << i + 1 << std::endl;
    }
  }

  // Ожидаем завершения всех потоков
  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  std::cout << "Process " << rank << ": All threads joined" << std::endl;

  // Собираем результаты со всех процессов
  for (int i = 0; i < 7; ++i) {
    std::vector<double> temp_result;

    // Проверяем, считал ли текущий процесс это значение
    bool calculated_locally = (i >= start_op && i < end_op);

    if (calculated_locally) {
      // Если да, то отправляем результат процессу 0
      if (rank != 0) {
        std::cout << "Process " << rank << ": Sending p" << i + 1 << " to process 0" << std::endl;
        world.send(0, i, p_results[i]);
      }
    } else if (rank == 0) {
      // Процесс 0 получает результаты от других процессов
      int source_rank = i / operations_per_process;
      if (remainder > 0) {
        if (i < remainder * (operations_per_process + 1)) {
          source_rank = i / (operations_per_process + 1);
        } else {
          source_rank = remainder + (i - remainder * (operations_per_process + 1)) / operations_per_process;
        }
      }

      if (source_rank != 0) {
        std::cout << "Process 0: Receiving p" << i + 1 << " from process " << source_rank << std::endl;
        world.recv(source_rank, i, temp_result);
        p_results[i] = temp_result;
      }
    }
  }

  // Только процесс с рангом 0 выполняет финальное вычисление
  if (rank == 0) {
    std::cout << "Process 0: Computing final result" << std::endl;

    // Извлекаем p1-p7 из p_results
    p1 = p_results[0];
    p2 = p_results[1];
    p3 = p_results[2];
    p4 = p_results[3];
    p5 = p_results[4];
    p6 = p_results[5];
    p7 = p_results[6];

    // Вычисляем финальные подматрицы C11, C12, C21, C22
    std::vector<double> c11 =
        AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
    std::vector<double> c12 = AddMatrices(p3, p5, half_size);
    std::vector<double> c21 = AddMatrices(p2, p4, half_size);
    std::vector<double> c22 =
        AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

    // Собираем финальный результат
    std::vector<double> result(size * size);
    MergeMatrix(result, c11, 0, 0, size);
    MergeMatrix(result, c12, 0, half_size, size);
    MergeMatrix(result, c21, half_size, 0, size);
    MergeMatrix(result, c22, half_size, half_size, size);

    std::cout << "Process 0: Final result computed" << std::endl;
    return result;
  } else {
    // Другие процессы возвращают пустой результат
    std::cout << "Process " << rank << ": Returning empty result" << std::endl;
    return std::vector<double>(size * size, 0.0);
  }
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