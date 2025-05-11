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
  std::cout << "Rank " << 0 << ", Thread " << std::this_thread::get_id() << ": Starting StrassenMultiply, size=" << size
            << ", num_threads=" << num_threads << std::endl;

  // Инициализация MPI, если еще не инициализирован
  boost::mpi::environment* env = nullptr;
  if (!boost::mpi::environment::initialized()) {
    std::cout << "Rank " << 0 << ", Thread " << std::this_thread::get_id() << ": Initializing MPI environment"
              << std::endl;
    static int argc = 0;
    static char** argv = nullptr;
    env = new boost::mpi::environment(argc, argv);
  }

  boost::mpi::communicator world;
  int rank = world.rank();
  int world_size = world.size();

  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
            << ": MPI initialized, world_size=" << world_size << std::endl;

  // Базовый случай или слишком маленькая матрица для распараллеливания
  if (size <= 32 || world_size <= 1) {
    std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Base case: size=" << size
              << ", world_size=" << world_size << ", using StandardMultiply" << std::endl;
    auto result = StandardMultiply(a, b, size);
    std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
              << ": Base case completed, result size=" << result.size() << std::endl;
    return result;
  }

  int half_size = size / 2;
  int half_size_squared = half_size * half_size;

  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Splitting matrices, size=" << size
            << ", half_size=" << half_size << ", half_size_squared=" << half_size_squared << std::endl;

  std::vector<double> a11(half_size_squared);
  std::vector<double> a12(half_size_squared);
  std::vector<double> a21(half_size_squared);
  std::vector<double> a22(half_size_squared);
  std::vector<double> b11(half_size_squared);
  std::vector<double> b12(half_size_squared);
  std::vector<double> b21(half_size_squared);
  std::vector<double> b22(half_size_squared);

  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
            << ": Splitting matrix A into a11, a12, a21, a22" << std::endl;
  SplitMatrix(a, a11, 0, 0, size);
  SplitMatrix(a, a12, 0, half_size, size);
  SplitMatrix(a, a21, half_size, 0, size);
  SplitMatrix(a, a22, half_size, half_size, size);

  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
            << ": Splitting matrix B into b11, b12, b21, b22" << std::endl;
  SplitMatrix(b, b11, 0, 0, size);
  SplitMatrix(b, b12, 0, half_size, size);
  SplitMatrix(b, b21, half_size, 0, size);
  SplitMatrix(b, b22, half_size, half_size, size);

  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Matrix splitting completed"
            << std::endl;

  std::vector<double> p1, p2, p3, p4, p5, p6, p7;

  // Распределяем задачи между процессами
  int tasks_per_process = 7 / world_size;
  int extra_tasks = 7 % world_size;
  int start_task = rank * tasks_per_process + std::min(rank, extra_tasks);
  int end_task = start_task + tasks_per_process + (rank < extra_tasks ? 1 : 0);

  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
            << ": Task distribution: tasks_per_process=" << tasks_per_process << ", extra_tasks=" << extra_tasks
            << ", start_task=" << start_task << ", end_task=" << end_task << std::endl;

  // Выполняем назначенные задачи
  for (int task_id = start_task; task_id < end_task; ++task_id) {
    std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Executing task ID=" << task_id
              << std::endl;
    switch (task_id) {
      case 0:
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
                  << ": Computing p1: (a11 + a22) * (b11 + b22)" << std::endl;
        p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size,
                              num_threads);
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": p1 computed, size=" << p1.size()
                  << std::endl;
        break;
      case 1:
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Computing p2: (a21 + a22) * b11"
                  << std::endl;
        p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads);
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": p2 computed, size=" << p2.size()
                  << std::endl;
        break;
      case 2:
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Computing p3: a11 * (b12 - b22)"
                  << std::endl;
        p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads);
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": p3 computed, size=" << p3.size()
                  << std::endl;
        break;
      case 3:
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Computing p4: a22 * (b21 - b11)"
                  << std::endl;
        p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads);
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": p4 computed, size=" << p4.size()
                  << std::endl;
        break;
      case 4:
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Computing p5: (a11 + a12) * b22"
                  << std::endl;
        p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads);
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": p5 computed, size=" << p5.size()
                  << std::endl;
        break;
      case 5:
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
                  << ": Computing p6: (a21 - a11) * (b11 + b12)" << std::endl;
        p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                              num_threads);
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": p6 computed, size=" << p6.size()
                  << std::endl;
        break;
      case 6:
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
                  << ": Computing p7: (a12 - a22) * (b21 + b22)" << std::endl;
        p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                              num_threads);
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": p7 computed, size=" << p7.size()
                  << std::endl;
        break;
    }
  }

  // Обмениваемся результатами между процессами
  if (world_size > 1) {
    std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Starting MPI broadcast"
              << std::endl;
    for (int task_id = 0; task_id < 7; ++task_id) {
      int owner = task_id % world_size;
      if (rank == owner) {
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Broadcasting p" << (task_id + 1)
                  << " for task ID=" << task_id << std::endl;
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
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Receiving p" << (task_id + 1)
                  << " for task ID=" << task_id << " from rank=" << owner << std::endl;
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
        std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Received p" << (task_id + 1)
                  << ", size="
                  << (task_id == 0   ? p1.size()
                      : task_id == 1 ? p2.size()
                      : task_id == 2 ? p3.size()
                      : task_id == 3 ? p4.size()
                      : task_id == 4 ? p5.size()
                      : task_id == 5 ? p6.size()
                                     : p7.size())
                  << std::endl;
      }
    }
    std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": MPI broadcast completed"
              << std::endl;
  }

  // Запускаем многопоточное выполнение задач, не выполненных через MPI
  std::vector<std::function<void()>> remaining_tasks;

  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Checking remaining tasks"
            << std::endl;
  if (p1.empty()) {
    std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Adding remaining task for p1"
              << std::endl;
    remaining_tasks.emplace_back([&]() {
      std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Computing p1 in thread"
                << std::endl;
      p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, num_threads);
      std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
                << ": p1 computed in thread, size=" << p1.size() << std::endl;
    });
  }

  if (p2.empty()) {
    std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Adding remaining task for p2"
              << std::endl;
    remaining_tasks.emplace_back([&]() {
      std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Computing p2 in thread"
                << std::endl;
      p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads);
      std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
                << ": p2 computed in thread, size=" << p2.size() << std::endl;
    });
  }

  if (p3.empty()) {
    std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Adding remaining task for p3"
              << std::endl;
    remaining_tasks.emplace_back([&]() {
      std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Computing p3 in thread"
                << std::endl;
      p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads);
      std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
                << ": p3 computed in thread, size=" << p3.size() << std::endl;
    });
  }

  if (p4.empty()) {
    std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Adding remaining task for p4"
              << std::endl;
    remaining_tasks.emplace_back([&]() {
      std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Computing p4 in thread"
                << std::endl;
      p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads);
      std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
                << ": p4 computed in thread, size=" << p4.size() << std::endl;
    });
  }

  if (p5.empty()) {
    std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Adding remaining task for p5"
              << std::endl;
    remaining_tasks.emplace_back([&]() {
      std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Computing p5 in thread"
                << std::endl;
      p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads);
      std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
                << ": p5 computed in thread, size=" << p5.size() << std::endl;
    });
  }

  if (p6.empty()) {
    std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Adding remaining task for p6"
              << std::endl;
    remaining_tasks.emplace_back([&]() {
      std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Computing p6 in thread"
                << std::endl;
      p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                            num_threads);
      std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
                << ": p6 computed in thread, size=" << p6.size() << std::endl;
    });
  }

  if (p7.empty()) {
    std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Adding remaining task for p7"
              << std::endl;
    remaining_tasks.emplace_back([&]() {
      std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Computing p7 in thread"
                << std::endl;
      p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                            num_threads);
      std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
                << ": p7 computed in thread, size=" << p7.size() << std::endl;
    });
  }

  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
            << ": Total remaining tasks=" << remaining_tasks.size() << std::endl;

  // Запускаем многопоточное выполнение оставшихся задач
  std::vector<std::thread> threads;
  threads.reserve(std::min(num_threads, static_cast<int>(remaining_tasks.size())));
  size_t task_index = 0;

  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
            << ": Launching threads, max threads=" << std::min(num_threads, static_cast<int>(remaining_tasks.size()))
            << std::endl;
  for (int i = 0; i < std::min(num_threads, static_cast<int>(remaining_tasks.size())); ++i) {
    if (task_index < remaining_tasks.size()) {
      std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
                << ": Starting thread for task index=" << task_index << std::endl;
      threads.emplace_back(remaining_tasks[task_index]);
      ++task_index;
    }
  }

  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
            << ": Executing remaining tasks sequentially" << std::endl;
  while (task_index < remaining_tasks.size()) {
    std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Executing task index=" << task_index
              << " sequentially" << std::endl;
    remaining_tasks[task_index]();
    ++task_index;
  }

  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Joining threads" << std::endl;
  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
      std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Thread joined" << std::endl;
    }
  }

  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Combining results" << std::endl;
  std::vector<double> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": c11 computed, size=" << c11.size()
            << std::endl;
  std::vector<double> c12 = AddMatrices(p3, p5, half_size);
  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": c12 computed, size=" << c12.size()
            << std::endl;
  std::vector<double> c21 = AddMatrices(p2, p4, half_size);
  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": c21 computed, size=" << c21.size()
            << std::endl;
  std::vector<double> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);
  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": c22 computed, size=" << c22.size()
            << std::endl;

  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Merging submatrices" << std::endl;
  std::vector<double> result(size * size);
  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);

  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id()
            << ": Result matrix created, size=" << result.size() << std::endl;

  // Очищаем созданный объект environment, если он был создан
  if (env != nullptr) {
    std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": Cleaning up MPI environment"
              << std::endl;
    delete env;
  }

  std::cout << "Rank " << rank << ", Thread " << std::this_thread::get_id() << ": StrassenMultiply completed"
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