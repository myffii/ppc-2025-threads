#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
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
  }

  // Рассылка матриц и размеров всем процессам
  boost::mpi::broadcast(world_, input_matrix_a_, 0);
  boost::mpi::broadcast(world_, input_matrix_b_, 0);
  boost::mpi::broadcast(world_, matrix_size_, 0);
  boost::mpi::broadcast(world_, original_size_, 0);

  // Инициализация output_matrix_ на всех процессах
  output_matrix_.resize(matrix_size_ * matrix_size_, 0.0);

  return true;
}

bool StrassenAll::ValidationImpl() {
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
  if (size <= 32) {
    if (world_.rank() == 0) {
      std::cout << "Process " << world_.rank() << ": StandardMultiply for size " << size << std::endl;
    }
    return StandardMultiply(a, b, size);
  }

  int half_size = size / 2;
  int half_size_squared = half_size * half_size;

  // Разделение матриц на подматрицы
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

  // Подматрицы для результатов
  std::vector<double> p1(half_size_squared, 0.0);
  std::vector<double> p2(half_size_squared, 0.0);
  std::vector<double> p3(half_size_squared, 0.0);
  std::vector<double> p4(half_size_squared, 0.0);
  std::vector<double> p5(half_size_squared, 0.0);
  std::vector<double> p6(half_size_squared, 0.0);
  std::vector<double> p7(half_size_squared, 0.0);

  // Распределение задач между процессами
  int rank = world_.rank();
  int num_procs = world_.size();
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int task_id = rank; task_id < 7; task_id += num_procs) {
    if (world_.rank() == 0) {
      std::cout << "Process " << rank << ": Assigned task " << task_id + 1 << std::endl;
    }
    switch (task_id) {
      case 0:
        threads.emplace_back([this, &p1, &a11, &a22, &b11, &b22, half_size, num_threads]() {
          if (world_.rank() == 0) {
            std::cout << "Process " << world_.rank() << ", Thread " << std::this_thread::get_id() << ": Computing p1"
                      << std::endl;
          }
          p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size,
                                num_threads);
        });
        break;
      case 1:
        threads.emplace_back([this, &p2, &a21, &a22, &b11, half_size, num_threads]() {
          if (world_.rank() == 0) {
            std::cout << "Process " << world_.rank() << ", Thread " << std::this_thread::get_id() << ": Computing p2"
                      << std::endl;
          }
          p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads);
        });
        break;
      case 2:
        threads.emplace_back([this, &p3, &a11, &b12, &b22, half_size, num_threads]() {
          if (world_.rank() == 0) {
            std::cout << "Process " << world_.rank() << ", Thread " << std::this_thread::get_id() << ": Computing p3"
                      << std::endl;
          }
          p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads);
        });
        break;
      case 3:
        threads.emplace_back([this, &p4, &a22, &b21, &b11, half_size, num_threads]() {
          if (world_.rank() == 0) {
            std::cout << "Process " << world_.rank() << ", Thread " << std::this_thread::get_id() << ": Computing p4"
                      << std::endl;
          }
          p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads);
        });
        break;
      case 4:
        threads.emplace_back([this, &p5, &a11, &a12, &b22, half_size, num_threads]() {
          if (world_.rank() == 0) {
            std::cout << "Process " << world_.rank() << ", Thread " << std::this_thread::get_id() << ": Computing p5"
                      << std::endl;
          }
          p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads);
        });
        break;
      case 5:
        threads.emplace_back([this, &p6, &a21, &a11, &b11, &b12, half_size, num_threads]() {
          if (world_.rank() == 0) {
            std::cout << "Process " << world_.rank() << ", Thread " << std::this_thread::get_id() << ": Computing p6"
                      << std::endl;
          }
          p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                                num_threads);
        });
        break;
      case 6:
        threads.emplace_back([this, &p7, &a12, &a22, &b21, &b22, half_size, num_threads]() {
          if (world_.rank() == 0) {
            std::cout << "Process " << world_.rank() << ", Thread " << std::this_thread::get_id() << ": Computing p7"
                      << std::endl;
          }
          p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                                num_threads);
        });
        break;
    }
  }

  // Ожидание завершения потоков
  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  // Сбор результатов от всех процессов
  std::vector<std::vector<double>> all_p(7, std::vector<double>(half_size_squared, 0.0));
  all_p[0] = p1;
  all_p[1] = p2;
  all_p[2] = p3;
  all_p[3] = p4;
  all_p[4] = p5;
  all_p[5] = p6;
  all_p[6] = p7;

  // Сбор каждой подматрицы отдельно
  for (int i = 0; i < 7; ++i) {
    std::vector<std::vector<double>> gathered_p_i;
    if (i % num_procs == rank) {
      boost::mpi::all_gather(world_, all_p[i], gathered_p_i);
    } else {
      boost::mpi::all_gather(world_, std::vector<double>(half_size_squared, 0.0), gathered_p_i);
    }
    // Копируем результат от ответственного процесса
    int responsible_proc = i % num_procs;
    all_p[i] = gathered_p_i[responsible_proc];
  }

  p1 = all_p[0];
  p2 = all_p[1];
  p3 = all_p[2];
  p4 = all_p[3];
  p5 = all_p[4];
  p6 = all_p[5];
  p7 = all_p[6];

  if (world_.rank() == 0) {
    std::cout << "Process " << world_.rank() << ": All results gathered, computing final matrix" << std::endl;
  }

  // Вычисление результирующих подматриц только на процессе 0
  std::vector<double> result(size * size, 0.0);
  if (world_.rank() == 0) {
    std::vector<double> c11 =
        AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
    std::vector<double> c12 = AddMatrices(p3, p5, half_size);
    std::vector<double> c21 = AddMatrices(p2, p4, half_size);
    std::vector<double> c22 =
        AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

    // Объединение подматриц
    MergeMatrix(result, c11, 0, 0, size);
    MergeMatrix(result, c12, 0, half_size, size);
    MergeMatrix(result, c21, half_size, 0, size);
    MergeMatrix(result, c22, half_size, half_size, size);

    std::cout << "Process " << world_.rank() << ": StrassenMultiply completed for size " << size << std::endl;
  }

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