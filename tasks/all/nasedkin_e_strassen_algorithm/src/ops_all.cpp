#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
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
  // Отключаем синхронизацию std::cout для ускорения вывода
  std::cout.sync_with_stdio(false);

  // Инициализация MPI, если еще не инициализирован
  boost::mpi::environment* env = nullptr;
  if (!boost::mpi::environment::initialized()) {
    std::cout << FormatDebugLog(0, "Initializing MPI environment");
    static int argc = 0;
    static char** argv = nullptr;
    env = new boost::mpi::environment(argc, argv);
  }

  boost::mpi::communicator world;
  int rank = world.rank();
  int world_size = world.size();

  std::cout << FormatDebugLog(rank, "Starting StrassenMultiply with matrix size=" + std::to_string(size) +
                                        ", num_threads=" + std::to_string(num_threads) +
                                        ", world_size=" + std::to_string(world_size));

  // Базовый случай или слишком маленькая матрица для распараллеливания
  if (size <= 32 || world_size <= 1) {
    std::cout << FormatDebugLog(rank, "Base case triggered: size=" + std::to_string(size) +
                                          ", world_size=" + std::to_string(world_size) + ". Using StandardMultiply.");
    auto result = StandardMultiply(a, b, size);
    std::cout << FormatDebugLog(rank, "Base case completed. Result size=" + std::to_string(result.size()));
    return result;
  }

  int half_size = size / 2;
  int half_size_squared = half_size * half_size;

  std::cout << FormatDebugLog(rank, "Splitting matrices: size=" + std::to_string(size) +
                                        ", half_size=" + std::to_string(half_size) +
                                        ", half_size_squared=" + std::to_string(half_size_squared));

  std::vector<double> a11(half_size_squared);
  std::vector<double> a12(half_size_squared);
  std::vector<double> a21(half_size_squared);
  std::vector<double> a22(half_size_squared);
  std::vector<double> b11(half_size_squared);
  std::vector<double> b12(half_size_squared);
  std::vector<double> b21(half_size_squared);
  std::vector<double> b22(half_size_squared);

  // На всех процессах разделяем матрицы на подматрицы
  std::cout << FormatDebugLog(rank, "Splitting matrix A into a11, a12, a21, a22");
  SplitMatrix(a, a11, 0, 0, size);
  SplitMatrix(a, a12, 0, half_size, size);
  SplitMatrix(a, a21, half_size, 0, size);
  SplitMatrix(a, a22, half_size, half_size, size);

  std::cout << FormatDebugLog(rank, "Splitting matrix B into b11, b12, b21, b22");
  SplitMatrix(b, b11, 0, 0, size);
  SplitMatrix(b, b12, 0, half_size, size);
  SplitMatrix(b, b21, half_size, 0, size);
  SplitMatrix(b, b22, half_size, half_size, size);

  std::cout << FormatDebugLog(rank, "Matrix splitting completed. Submatrix size=" + std::to_string(half_size_squared));

  std::vector<double> p1, p2, p3, p4, p5, p6, p7;

  // Распределяем задачи между процессами
  int tasks_per_process = 7 / world_size;
  int extra_tasks = 7 % world_size;

  int start_task = rank * tasks_per_process + std::min(rank, extra_tasks);
  int end_task = start_task + tasks_per_process + (rank < extra_tasks ? 1 : 0);

  std::cout << FormatDebugLog(rank, "Task distribution: tasks_per_process=" + std::to_string(tasks_per_process) +
                                        ", extra_tasks=" + std::to_string(extra_tasks) + ", start_task=" +
                                        std::to_string(start_task) + ", end_task=" + std::to_string(end_task));

  // Выполняем назначенные задачи
  for (int task_id = start_task; task_id < end_task; ++task_id) {
    std::cout << FormatDebugLog(rank, "Executing task ID=" + std::to_string(task_id));
    switch (task_id) {
      case 0:
        std::cout << FormatDebugLog(rank, "Computing p1: (a11 + a22) * (b11 + b22)");
        p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size,
                              num_threads);
        std::cout << FormatDebugLog(rank, "p1 computed. Size=" + std::to_string(p1.size()));
        break;
      case 1:
        std::cout << FormatDebugLog(rank, "Computing p2: (a21 + a22) * b11");
        p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads);
        std::cout << FormatDebugLog(rank, "p2 computed. Size=" + std::to_string(p2.size()));
        break;
      case 2:
        std::cout << FormatDebugLog(rank, "Computing p3: a11 * (b12 - b22)");
        p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads);
        std::cout << FormatDebugLog(rank, "p3 computed. Size=" + std::to_string(p3.size()));
        break;
      case 3:
        std::cout << FormatDebugLog(rank, "Computing p4: a22 * (b21 - b11)");
        p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads);
        std::cout << FormatDebugLog(rank, "p4 computed. Size=" + std::to_string(p4.size()));
        break;
      case 4:
        std::cout << FormatDebugLog(rank, "Computing p5: (a11 + a12) * b22");
        p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads);
        std::cout << FormatDebugLog(rank, "p5 computed. Size=" + std::to_string(p5.size()));
        break;
      case 5:
        std::cout << FormatDebugLog(rank, "Computing p6: (a21 - a11) * (b11 + b12)");
        p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                              num_threads);
        std::cout << FormatDebugLog(rank, "p6 computed. Size=" + std::to_string(p6.size()));
        break;
      case 6:
        std::cout << FormatDebugLog(rank, "Computing p7: (a12 - a22) * (b21 + b22)");
        p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                              num_threads);
        std::cout << FormatDebugLog(rank, "p7 computed. Size=" + std::to_string(p7.size()));
        break;
    }
  }

  // Обмениваемся результатами между процессами
  if (world_size > 1) {
    std::cout << FormatDebugLog(rank, "Starting MPI broadcast for results");
    for (int task_id = 0; task_id < 7; ++task_id) {
      int owner = task_id % world_size;
      if (rank == owner) {
        std::cout << FormatDebugLog(rank, "Broadcasting result for task ID=" + std::to_string(task_id) + " (p" +
                                              std::to_string(task_id + 1) + ")");
        switch (task_id) {
          case 0:
            boost::mpi::broadcast(world, p1, owner);
            break;
          case 1:
            boost::mpi::broadcast(world, p2, owner);
            break;
          case 2:
            boost::mpi::broadcast(world, p3, owner);
            break;
          case 3:
            boost::mpi::broadcast(world, p4, owner);
            break;
          case 4:
            boost::mpi::broadcast(world, p5, owner);
            break;
          case 5:
            boost::mpi::broadcast(world, p6, owner);
            break;
          case 6:
            boost::mpi::broadcast(world, p7, owner);
            break;
        }
      } else {
        std::cout << FormatDebugLog(rank, "Receiving result for task ID=" + std::to_string(task_id) + " (p" +
                                              std::to_string(task_id + 1) + ") from rank=" + std::to_string(owner));
        switch (task_id) {
          case 0:
            boost::mpi::broadcast(world, p1, owner);
            break;
          case 1:
            boost::mpi::broadcast(world, p2, owner);
            break;
          case 2:
            boost::mpi::broadcast(world, p3, owner);
            break;
          case 3:
            boost::mpi::broadcast(world, p4, owner);
            break;
          case 4:
            boost::mpi::broadcast(world, p5, owner);
            break;
          case 5:
            boost::mpi::broadcast(world, p6, owner);
            break;
          case 6:
            boost::mpi::broadcast(world, p7, owner);
            break;
        }
        std::cout << FormatDebugLog(rank, "Received result for task ID=" + std::to_string(task_id) + ". Size=" +
                                              std::to_string((task_id == 0   ? p1
                                                              : task_id == 1 ? p2
                                                              : task_id == 2 ? p3
                                                              : task_id == 3 ? p4
                                                              : task_id == 4 ? p5
                                                              : task_id == 5 ? p6
                                                                             : p7)
                                                                 .size()));
      }
    }
    std::cout << FormatDebugLog(rank, "MPI broadcast completed");
  }

  // Запускаем многопоточное выполнение задач, не выполненных через MPI
  std::vector<std::function<void()>> remaining_tasks;

  std::cout << FormatDebugLog(rank, "Checking for remaining tasks");
  if (p1.empty()) {
    std::cout << FormatDebugLog(rank, "Adding remaining task for p1");
    remaining_tasks.emplace_back([&]() {
      std::cout << FormatDebugLog(rank, "Computing remaining p1 in thread");
      p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, num_threads);
      std::cout << FormatDebugLog(rank, "Remaining p1 computed. Size=" + std::to_string(p1.size()));
    });
  }

  if (p2.empty()) {
    std::cout << FormatDebugLog(rank, "Adding remaining task for p2");
    remaining_tasks.emplace_back([&]() {
      std::cout << FormatDebugLog(rank, "Computing remaining p2 in thread");
      p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads);
      std::cout << FormatDebugLog(rank, "Remaining p2 computed. Size=" + std::to_string(p2.size()));
    });
  }

  if (p3.empty()) {
    std::cout << FormatDebugLog(rank, "Adding remaining task for p3");
    remaining_tasks.emplace_back([&]() {
      std::cout << FormatDebugLog(rank, "Computing remaining p3 in thread");
      p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads);
      std::cout << FormatDebugLog(rank, "Remaining p3 computed. Size=" + std::to_string(p3.size()));
    });
  }

  if (p4.empty()) {
    std::cout << FormatDebugLog(rank, "Adding remaining task for p4");
    remaining_tasks.emplace_back([&]() {
      std::cout << FormatDebugLog(rank, "Computing remaining p4 in thread");
      p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads);
      std::cout << FormatDebugLog(rank, "Remaining p4 computed. Size=" + std::to_string(p4.size()));
    });
  }

  if (p5.empty()) {
    std::cout << FormatDebugLog(rank, "Adding remaining task for p5");
    remaining_tasks.emplace_back([&]() {
      std::cout << FormatDebugLog(rank, "Computing remaining p5 in thread");
      p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads);
      std::cout << FormatDebugLog(rank, "Remaining p5 computed. Size=" + std::to_string(p5.size()));
    });
  }

  if (p6.empty()) {
    std::cout << FormatDebugLog(rank, "Adding remaining task for p6");
    remaining_tasks.emplace_back([&]() {
      std::cout << FormatDebugLog(rank, "Computing remaining p6 in thread");
      p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                            num_threads);
      std::cout << FormatDebugLog(rank, "Remaining p6 computed. Size=" + std::to_string(p6.size()));
    });
  }

  if (p7.empty()) {
    std::cout << FormatDebugLog(rank, "Adding remaining task for p7");
    remaining_tasks.emplace_back([&]() {
      std::cout << FormatDebugLog(rank, "Computing remaining p7 in thread");
      p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                            num_threads);
      std::cout << FormatDebugLog(rank, "Remaining p7 computed. Size=" + std::to_string(p7.size()));
    });
  }

  std::cout << FormatDebugLog(rank, "Total remaining tasks: " + std::to_string(remaining_tasks.size()));

  // Запускаем многопоточное выполнение оставшихся задач
  std::vector<std::thread> threads;
  threads.reserve(std::min(num_threads, static_cast<int>(remaining_tasks.size())));
  size_t task_index = 0;

  std::cout << FormatDebugLog(rank,
                              "Launching threads for remaining tasks. Max threads=" +
                                  std::to_string(std::min(num_threads, static_cast<int>(remaining_tasks.size()))));
  for (int i = 0; i < std::min(num_threads, static_cast<int>(remaining_tasks.size())); ++i) {
    if (task_index < remaining_tasks.size()) {
      std::cout << FormatDebugLog(rank, "Starting thread for task index=" + std::to_string(task_index));
      threads.emplace_back(remaining_tasks[task_index]);
      ++task_index;
    }
  }

  std::cout << FormatDebugLog(rank, "Executing remaining tasks sequentially");
  while (task_index < remaining_tasks.size()) {
    std::cout << FormatDebugLog(rank, "Executing task index=" + std::to_string(task_index) + " sequentially");
    remaining_tasks[task_index]();
    ++task_index;
  }

  std::cout << FormatDebugLog(rank, "Joining threads");
  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
      std::cout << FormatDebugLog(rank, "Thread joined");
    }
  }

  std::cout << FormatDebugLog(rank, "Combining results for final matrix");
  // Комбинируем результаты для получения финальной матрицы
  std::cout << FormatDebugLog(rank, "Computing c11: (p1 + p4 - p5 + p7)");
  std::vector<double> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  std::cout << FormatDebugLog(rank, "Computing c12: (p3 + p5)");
  std::vector<double> c12 = AddMatrices(p3, p5, half_size);
  std::cout << FormatDebugLog(rank, "Computing c21: (p2 + p4)");
  std::vector<double> c21 = AddMatrices(p2, p4, half_size);
  std::cout << FormatDebugLog(rank, "Computing c22: (p1 + p3 - p2 + p6)");
  std::vector<double> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

  std::cout << FormatDebugLog(rank, "Merging submatrices into result");
  std::vector<double> result(size * size);
  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);

  std::cout << FormatDebugLog(rank, "Result matrix created. Size=" + std::to_string(result.size()));

  // Очищаем созданный объект environment, если он был создан
  if (env != nullptr) {
    std::cout << FormatDebugLog(rank, "Cleaning up MPI environment");
    delete env;
  }

  std::cout << FormatDebugLog(rank, "StrassenMultiply completed");
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