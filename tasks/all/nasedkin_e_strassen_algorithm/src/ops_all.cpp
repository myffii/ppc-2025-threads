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

#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"
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
  int rank = world.rank();

  // Синхронизируем входные матрицы
  boost::mpi::broadcast(world, input_matrix_a_, 0);
  boost::mpi::broadcast(world, input_matrix_b_, 0);

  std::cout << "[DEBUG] Process " << rank << ": After input synchronization: "
            << "input_matrix_a_[0] = " << (input_matrix_a_.empty() ? 0.0 : input_matrix_a_[0])
            << ", input_matrix_b_[0] = " << (input_matrix_b_.empty() ? 0.0 : input_matrix_b_[0]) << std::endl;

  int num_threads = std::min(16, ppc::util::GetPPCNumThreads());  // Оставлено без изменений
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
  std::cout << "[DEBUG] StandardMultiply: size = " << size << ", a[0] = " << (a.empty() ? 0.0 : a[0])
            << ", b[0] = " << (b.empty() ? 0.0 : b[0]) << std::endl;
  std::vector<double> result(size * size, 0.0);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        result[(i * size) + j] += a[(i * size) + k] * b[(k * size) + j];
      }
    }
  }
  std::cout << "[DEBUG] StandardMultiply: result[0] = " << result[0] << std::endl;
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
  int world_size = world.size();

  std::cout << "[DEBUG] Process " << rank << ": Starting StrassenMultiply with size = " << size
            << ", a[0] = " << (a.empty() ? 0.0 : a[0]) << ", b[0] = " << (b.empty() ? 0.0 : b[0])
            << ", num_threads = " << num_threads << std::endl;

  // Базовый случай
  if (size <= 32 || world_size <= 1) {
    std::cout << "[DEBUG] Process " << rank << ": Using StandardMultiply for size = " << size << std::endl;
    return StandardMultiply(a, b, size);
  }

  int half_size = size / 2;
  int half_size_squared = half_size * half_size;

  // Инициализация подматриц с резервированием памяти
  std::vector<double> a11(half_size_squared);
  a11.reserve(half_size_squared);
  std::vector<double> a12(half_size_squared);
  a12.reserve(half_size_squared);
  std::vector<double> a21(half_size_squared);
  a21.reserve(half_size_squared);
  std::vector<double> a22(half_size_squared);
  a22.reserve(half_size_squared);
  std::vector<double> b11(half_size_squared);
  b11.reserve(half_size_squared);
  std::vector<double> b12(half_size_squared);
  b12.reserve(half_size_squared);
  std::vector<double> b21(half_size_squared);
  b21.reserve(half_size_squared);
  std::vector<double> b22(half_size_squared);
  b22.reserve(half_size_squared);

  // Разделение матриц
  SplitMatrix(a, a11, 0, 0, size);
  SplitMatrix(a, a12, 0, half_size, size);
  SplitMatrix(a, a21, half_size, 0, size);
  SplitMatrix(a, a22, half_size, half_size, size);
  SplitMatrix(b, b11, 0, 0, size);
  SplitMatrix(b, b12, 0, half_size, size);
  SplitMatrix(b, b21, half_size, 0, size);
  SplitMatrix(b, b22, half_size, half_size, size);

  std::cout << "[DEBUG] Process " << rank << ": Submatrices sizes: a11 = " << a11.size() << ", a12 = " << a12.size()
            << ", a21 = " << a21.size() << ", a22 = " << a22.size() << ", b11 = " << b11.size()
            << ", b12 = " << b12.size() << ", b21 = " << b21.size() << ", b22 = " << b22.size() << std::endl;

  // Инициализация p1–p7
  std::vector<double> p1(half_size_squared, 0.0);
  std::vector<double> p2(half_size_squared, 0.0);
  std::vector<double> p3(half_size_squared, 0.0);
  std::vector<double> p4(half_size_squared, 0.0);
  std::vector<double> p5(half_size_squared, 0.0);
  std::vector<double> p6(half_size_squared, 0.0);
  std::vector<double> p7(half_size_squared, 0.0);

  // Определение задач
  std::vector<std::function<void()>> tasks;
  tasks.emplace_back([a11 = a11, a22 = a22, b11 = b11, b22 = b22, half_size, num_threads, &p1]() {
    p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, num_threads);
  });
  tasks.emplace_back([a21 = a21, a22 = a22, b11 = b11, half_size, num_threads, &p2]() {
    p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads);
  });
  tasks.emplace_back([a11 = a11, b12 = b12, b22 = b22, half_size, num_threads, &p3]() {
    p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size),

                          half_size, num_threads);
  });
  tasks.emplace_back([a22 = a22, b21 = b21, b11 = b11, half_size, num_threads, &p4]() {
    p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads);
  });
  tasks.emplace_back([a11 = a11, a12 = a12, b22 = b22, half_size, num_threads, &p5]() {
    p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads);
  });
  tasks.emplace_back([a21 = a21, a11 = a11, b11 = b11, b12 = b12, half_size, num_threads, &p6]() {
    p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                          num_threads);
  });
  tasks.emplace_back([a12 = a12, a22 = a22, b21 = b21, b22 = b22, half_size, num_threads, &p7]() {
    p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                          num_threads);
  });

  // Распределение задач по процессам
  std::vector<std::function<void()>> my_tasks;
  std::vector<std::size_t> my_task_indices;
  if (rank < static_cast<int>(tasks.size()) && rank < world_size) {
    if (rank < std::min(world_size, static_cast<int>(tasks.size()))) {
      my_tasks.push_back(tasks[rank]);
      my_task_indices.push_back(rank);
    }
  }

  std::cout << "[DEBUG] Process " << rank << " assigned tasks: ";
  if (my_task_indices.empty()) {
    std::cout << "none";
  } else {
    for (std::size_t idx : my_task_indices) {
      std::cout << "p" << (idx + 1) << " ";
    }
  }
  std::cout << std::endl;

  // Выполнение назначенных задач
  for (auto& task : my_tasks) {
    task();
  }

  // Синхронизация результатов
  world.barrier();
  boost::mpi::broadcast(world, p1, 0);
  if (world_size > 1) {
    world.barrier();
    boost::mpi::broadcast(world, p2, 1);
  }
  if (world_size > 2) {
    world.barrier();
    boost::mpi::broadcast(world, p3, 2);
  }
  if (world_size > 3) {
    world.barrier();
    boost::mpi::broadcast(world, p4, 3);
  }

  std::cout << "[DEBUG] Process " << rank << " after broadcast of MPI tasks:" << std::endl;
  std::cout << "p1[0] = " << (p1.empty() ? 0.0 : p1[0]) << ", size = " << p1.size() << std::endl;
  std::cout << "p2[0] = " << (p2.empty() ? 0.0 : p2[0]) << ", size = " << p2.size() << std::endl;
  std::cout << "p3[0] = " << (p3.empty() ? 0.0 : p3[0]) << ", size = " << p3.size() << std::endl;
  std::cout << "p4[0] = " << (p4.empty() ? 0.0 : p4[0]) << ", size = " << p4.size() << std::endl;
  std::cout << "p5[0] = " << (p5.empty() ? 0.0 : p5[0]) << ", size = " << p5.size() << std::endl;
  std::cout << "p6[0] = " << (p6.empty() ? 0.0 : p6[0]) << ", size = " << p6.size() << std::endl;
  std::cout << "p7[0] = " << (p7.empty() ? 0.0 : p7[0]) << ", size = " << p7.size() << std::endl;

  // Оставшиеся задачи на процессе 0
  std::vector<std::function<void()>> remaining_tasks;
  std::vector<std::size_t> remaining_task_indices;
  if (rank == 0) {
    for (std::size_t i = world_size; i < tasks.size(); ++i) {
      remaining_tasks.push_back(tasks[i]);
      remaining_task_indices.push_back(i);
    }
  }

  if (rank == 0) {
    std::cout << "[DEBUG] Process " << rank << " running remaining tasks (multithreaded): ";
    if (remaining_task_indices.empty()) {
      std::cout << "none";
    } else {
      for (std::size_t idx : remaining_task_indices) {
        std::cout << "p" << (idx + 1) << " ";
      }
    }
    std::cout << std::endl;
  }

  // Выполнение оставшихся задач в потоках
  if (rank == 0 && !remaining_tasks.empty()) {
    std::vector<std::thread> threads;
    threads.reserve(std::min(num_threads, static_cast<int>(remaining_tasks.size())));
    std::size_t task_index = 0;

    std::cout << "[DEBUG] Process " << rank << ": Starting multithreaded tasks with " << num_threads << " threads"
              << std::endl;

    for (int i = 0; i < std::min(num_threads, static_cast<int>(remaining_tasks.size())); ++i) {
      if (task_index < remaining_tasks.size()) {
        threads.emplace_back(remaining_tasks[task_index]);
        ++task_index;
      }
    }

    while (task_index < remaining_tasks.size()) {
      remaining_tasks[task_index]();
      ++task_index;
    }

    for (auto& thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }

    std::cout << "[DEBUG] Process " << rank << ": Completed multithreaded tasks" << std::endl;
  }

  // Синхронизация оставшихся результатов
  if (rank == 0) {
    if (world_size <= 2) {
      world.barrier();
      boost::mpi::broadcast(world, p3, 0);
    }
    if (world_size <= 3) {
      world.barrier();
      boost::mpi::broadcast(world, p4, 0);
    }
    world.barrier();
    boost::mpi::broadcast(world, p5, 0);
    world.barrier();
    boost::mpi::broadcast(world, p6, 0);
    world.barrier();
    boost::mpi::broadcast(world, p7, 0);
  } else {
    if (world_size <= 2) {
      world.barrier();
      boost::mpi::broadcast(world, p3, 0);
    }
    if (world_size <= 3) {
      world.barrier();
      boost::mpi::broadcast(world, p4, 0);
    }
    world.barrier();
    boost::mpi::broadcast(world, p5, 0);
    world.barrier();
    boost::mpi::broadcast(world, p6, 0);
    world.barrier();
    boost::mpi::broadcast(world, p7, 0);
  }

  std::cout << "[DEBUG] Process " << rank << " after broadcast of remaining tasks:" << std::endl;
  std::cout << "p1[0] = " << (p1.empty() ? 0.0 : p1[0]) << ", size = " << p1.size() << std::endl;
  std::cout << "p2[0] = " << (p2.empty() ? 0.0 : p2[0]) << ", size = " << p2.size() << std::endl;
  std::cout << "p3[0] = " << (p3.empty() ? 0.0 : p3[0]) << ", size = " << p3.size() << std::endl;
  std::cout << "p4[0] = " << (p4.empty() ? 0.0 : p4[0]) << ", size = " << p4.size() << std::endl;
  std::cout << "p5[0] = " << (p5.empty() ? 0.0 : p5[0]) << ", size = " << p5.size() << std::endl;
  std::cout << "p6[0] = " << (p6.empty() ? 0.0 : p6[0]) << ", size = " << p6.size() << std::endl;
  std::cout << "p7[0] = " << (p7.empty() ? 0.0 : p7[0]) << ", size = " << p7.size() << std::endl;

  // Комбинирование результатов
  std::vector<double> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  std::vector<double> c12 = AddMatrices(p3, p5, half_size);
  std::vector<double> c21 = AddMatrices(p2, p4, half_size);
  std::vector<double> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

  std::cout << "[DEBUG] Process " << rank << " intermediate submatrices:" << std::endl;
  std::cout << "c11[0] = " << (c11.empty() ? 0.0 : c11[0]) << ", size = " << c11.size() << std::endl;
  std::cout << "c12[0] = " << (c12.empty() ? 0.0 : c12[0]) << ", size = " << c12.size() << std::endl;
  std::cout << "c21[0] = " << (c21.empty() ? 0.0 : c21[0]) << ", size = " << c21.size() << std::endl;
  std::cout << "c22[0] = " << (c22.empty() ? 0.0 : c22[0]) << ", size = " << c22.size() << std::endl;

  std::vector<double> result(size * size);
  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);

  std::cout << "[DEBUG] Process " << rank << ": Final result: result[0] = " << result[0] << ", size = " << result.size()
            << std::endl;

  return result;
}

void StrassenAll::SplitMatrix(const std::vector<double>& parent, std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  if (row_start + child_size > parent_size || col_start + child_size > parent_size) {
    std::cerr << "[ERROR] Process " << boost::mpi::communicator().rank()
              << ": Invalid split boundaries: row_start=" << row_start << ", col_start=" << col_start
              << ", child_size=" << child_size << ", parent_size=" << parent_size << std::endl;
    return;
  }
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(parent.begin() + (row_start + i) * parent_size + col_start,
                      parent.begin() + (row_start + i) * parent_size + col_start + child_size,
                      child.begin() + i * child_size);
  }
}

void StrassenAll::MergeMatrix(std::vector<double>& parent, const std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  if (row_start + child_size > parent_size || col_start + child_size > parent_size) {
    std::cerr << "[ERROR] Process " << boost::mpi::communicator().rank()
              << ": Invalid merge boundaries: row_start=" << row_start << ", col_start=" << col_start
              << ", child_size=" << child_size << ", parent_size=" << parent_size << std::endl;
    return;
  }
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(child.begin() + i * child_size, child.begin() + (i + 1) * child_size,
                      parent.begin() + (row_start + i) * parent_size + col_start);
  }
}

}  // namespace nasedkin_e_strassen_algorithm_all