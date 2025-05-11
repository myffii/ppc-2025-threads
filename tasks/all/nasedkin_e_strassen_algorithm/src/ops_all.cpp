#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
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

  // Базовый случай - используем стандартное умножение
  if (size <= 32) {
    return StandardMultiply(a, b, size);
  }

  int half_size = size / 2;
  int half_size_squared = half_size * half_size;

  // Подматрицы
  std::vector<double> a11(half_size_squared), a12(half_size_squared), a21(half_size_squared), a22(half_size_squared),
      b11(half_size_squared), b12(half_size_squared), b21(half_size_squared), b22(half_size_squared);

  // Только главный процесс разбивает матрицы
  if (world.rank() == 0) {
    SplitMatrix(a, a11, 0, 0, size);
    SplitMatrix(a, a12, 0, half_size, size);
    SplitMatrix(a, a21, half_size, 0, size);
    SplitMatrix(a, a22, half_size, half_size, size);

    SplitMatrix(b, b11, 0, 0, size);
    SplitMatrix(b, b12, 0, half_size, size);
    SplitMatrix(b, b21, half_size, 0, size);
    SplitMatrix(b, b22, half_size, half_size, size);
  }

  // Временные матрицы для промежуточных результатов
  std::vector<double> p1(half_size_squared), p2(half_size_squared), p3(half_size_squared), p4(half_size_squared),
      p5(half_size_squared), p6(half_size_squared), p7(half_size_squared);

  // Если есть другие MPI процессы
  if (world.size() > 1) {
    if (world.rank() == 0) {
      // Главный процесс распределяет задачи
      std::vector<std::thread> threads;
      int tasks_remaining = 7;

      // Отправляем задачи другим процессам
      for (int i = 1; i < world.size() && tasks_remaining > 0; ++i) {
        int task_num = 7 - tasks_remaining;

        switch (task_num) {
          case 0:  // P1 = (A11+A22)*(B11+B22)
            world.send(i, 0, AddMatrices(a11, a22, half_size));
            world.send(i, 0, AddMatrices(b11, b22, half_size));
            break;
          case 1:  // P2 = (A21+A22)*B11
            world.send(i, 0, AddMatrices(a21, a22, half_size));
            world.send(i, 0, b11);
            break;
          case 2:  // P3 = A11*(B12-B22)
            world.send(i, 0, a11);
            world.send(i, 0, SubtractMatrices(b12, b22, half_size));
            break;
          case 3:  // P4 = A22*(B21-B11)
            world.send(i, 0, a22);
            world.send(i, 0, SubtractMatrices(b21, b11, half_size));
            break;
          case 4:  // P5 = (A11+A12)*B22
            world.send(i, 0, AddMatrices(a11, a12, half_size));
            world.send(i, 0, b22);
            break;
          case 5:  // P6 = (A21-A11)*(B11+B12)
            world.send(i, 0, SubtractMatrices(a21, a11, half_size));
            world.send(i, 0, AddMatrices(b11, b12, half_size));
            break;
          case 6:  // P7 = (A12-A22)*(B21+B22)
            world.send(i, 0, SubtractMatrices(a12, a22, half_size));
            world.send(i, 0, AddMatrices(b21, b22, half_size));
            break;
        }
        tasks_remaining--;
      }

      // Локально выполняем оставшиеся задачи с STL потоками
      std::vector<std::function<void()>> tasks;
      if (tasks_remaining > 0)
        tasks.emplace_back([&]() {
          p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size,
                                num_threads);
        });
      if (tasks_remaining > 1)
        tasks.emplace_back(
            [&]() { p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads); });
      if (tasks_remaining > 2)
        tasks.emplace_back(
            [&]() { p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads); });
      if (tasks_remaining > 3)
        tasks.emplace_back(
            [&]() { p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads); });
      if (tasks_remaining > 4)
        tasks.emplace_back(
            [&]() { p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads); });
      if (tasks_remaining > 5)
        tasks.emplace_back([&]() {
          p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                                num_threads);
        });
      if (tasks_remaining > 6)
        tasks.emplace_back([&]() {
          p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                                num_threads);
        });

      // Запускаем потоки для оставшихся задач
      for (size_t i = 0; i < tasks.size(); ++i) {
        if (i < num_threads) {
          threads.emplace_back(tasks[i]);
        } else {
          tasks[i]();  // Выполняем синхронно, если потоки закончились
        }
      }

      // Ждем завершения потоков
      for (auto& t : threads) {
        if (t.joinable()) t.join();
      }

      // Получаем результаты от других процессов
      for (int i = 1; i < world.size() && (7 - tasks_remaining) > 0; ++i) {
        int task_num = 7 - tasks_remaining;
        std::vector<double> result;
        world.recv(i, 0, result);

        switch (task_num) {
          case 0:
            p1 = std::move(result);
            break;
          case 1:
            p2 = std::move(result);
            break;
          case 2:
            p3 = std::move(result);
            break;
          case 3:
            p4 = std::move(result);
            break;
          case 4:
            p5 = std::move(result);
            break;
          case 5:
            p6 = std::move(result);
            break;
          case 6:
            p7 = std::move(result);
            break;
        }
        tasks_remaining--;
      }
    } else {
      // Рабочие процессы
      std::vector<double> local_a, local_b;
      world.recv(0, 0, local_a);
      world.recv(0, 0, local_b);

      auto result = StrassenMultiply(local_a, local_b, half_size, num_threads);
      world.send(0, 0, result);
      return {};
    }
  } else {
    // Только STL потоки, если нет других MPI процессов
    std::vector<std::thread> threads;
    std::vector<std::function<void()>> tasks = {
        [&]() {
          p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size,
                                num_threads);
        },
        [&]() { p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads); },
        [&]() { p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads); },
        [&]() { p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads); },
        [&]() { p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads); },
        [&]() {
          p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                                num_threads);
        },
        [&]() {
          p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                                num_threads);
        }};

    // Запускаем потоки
    for (size_t i = 0; i < tasks.size(); ++i) {
      if (i < static_cast<size_t>(num_threads)) {
        threads.emplace_back(tasks[i]);
      } else {
        tasks[i]();  // Выполняем синхронно, если потоки закончились
      }
    }

    // Ждем завершения потоков
    for (auto& t : threads) {
      if (t.joinable()) t.join();
    }
  }

  // Только главный процесс собирает результат
  if (world.rank() == 0) {
    // Вычисляем подматрицы результата
    std::vector<double> c11 =
        AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
    std::vector<double> c12 = AddMatrices(p3, p5, half_size);
    std::vector<double> c21 = AddMatrices(p2, p4, half_size);
    std::vector<double> c22 =
        AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

    // Собираем итоговую матрицу
    std::vector<double> result(size * size);
    MergeMatrix(result, c11, 0, 0, size);
    MergeMatrix(result, c12, 0, half_size, size);
    MergeMatrix(result, c21, half_size, 0, size);
    MergeMatrix(result, c22, half_size, half_size, size);

    return result;
  }

  return {};
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