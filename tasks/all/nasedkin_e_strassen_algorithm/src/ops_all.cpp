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

  // Распределяем 7 произведений между MPI-процессами
  std::vector<std::function<void()>> tasks;
  if (world.rank() == 0) {
    tasks.push_back([&]() {
      p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, world);
    });
  } else if (world.rank() == 1) {
    tasks.push_back([&]() { p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, world); });
  } else if (world.rank() == 2) {
    tasks.push_back([&]() { p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, world); });
  } else if (world.rank() == 3) {
    tasks.push_back([&]() { p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, world); });
  } else {
    // Дополнительные процессы могут выполнять оставшиеся задачи
    int task_id = world.rank() % 3 + 4;
    if (task_id == 4) {
      tasks.push_back([&]() { p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, world); });
    } else if (task_id == 5) {
      tasks.push_back([&]() {
        p6 =
            StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size, world);
      });
    } else if (task_id == 6) {
      tasks.push_back([&]() {
        p7 =
            StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size, world);
      });
    }
  }

  std::cout << "[DEBUG] StrassenMultiply: Rank = " << world.rank() << ", Tasks assigned: " << tasks.size() << std::endl;

  // Параллелим задачи внутри процесса с помощью потоков
  int num_threads = ppc::util::GetPPCNumThreads();
  std::cout << "[DEBUG] StrassenMultiply: Rank = " << world.rank() << ", Using " << num_threads << " threads"
            << std::endl;

  std::vector<std::thread> threads;
  for (size_t i = 0; i < tasks.size(); ++i) {
    if (i < static_cast<size_t>(num_threads)) {
      threads.emplace_back(tasks[i]);
    } else {
      tasks[i]();  // Выполняем последовательно, если потоков не хватает
    }
  }

  for (auto& t : threads) {
    t.join();
  }

  std::cout << "[DEBUG] StrassenMultiply: Rank = " << world.rank() << ", Threads joined" << std::endl;

  // Собираем результаты через MPI
  if (world.rank() == 0) {
    // Главный процесс уже имеет p1
    if (world.size() > 1) {
      world.recv(1, 1, p2.data(), half_size_squared);
      std::cout << "[DEBUG] StrassenMultiply: Rank 0 received p2" << std::endl;
    }
    if (world.size() > 2) {
      world.recv(2, 2, p3.data(), half_size_squared);
      std::cout << "[DEBUG] StrassenMultiply: Rank 0 received p3" << std::endl;
    }
    if (world.size() > 3) {
      world.recv(3, 3, p4.data(), half_size_squared);
      std::cout << "[DEBUG] StrassenMultiply: Rank 0 received p4" << std::endl;
    }
    if (world.size() > 4) {
      world.recv(4, 4, p5.data(), half_size_squared);
      std::cout << "[DEBUG] StrassenMultiply: Rank 0 received p5" << std::endl;
    }
    if (world.size() > 5) {
      world.recv(5, 5, p6.data(), half_size_squared);
      std::cout << "[DEBUG] StrassenMultiply: Rank 0 received p6" << std::endl;
    }
    if (world.size() > 6) {
      world.recv(6, 6, p7.data(), half_size_squared);
      std::cout << "[DEBUG] StrassenMultiply: Rank 0 received p7" << std::endl;
    }
  } else {
    if (world.rank() == 1 && !p2.empty()) {
      world.send(0, 1, p2.data(), half_size_squared);
      std::cout << "[DEBUG] StrassenMultiply: Rank 1 sent p2" << std::endl;
    } else if (world.rank() == 2 && !p3.empty()) {
      world.send(0, 2, p3.data(), half_size_squared);
      std::cout << "[DEBUG] StrassenMultiply: Rank 2 sent p3" << std::endl;
    } else if (world.rank() == 3 && !p4.empty()) {
      world.send(0, 3, p4.data(), half_size_squared);
      std::cout << "[DEBUG] StrassenMultiply: Rank 3 sent p4" << std::endl;
    } else if (world.rank() == 4 && !p5.empty()) {
      world.send(0, 4, p5.data(), half_size_squared);
      std::cout << "[DEBUG] StrassenMultiply: Rank 4 sent p5" << std::endl;
    } else if (world.rank() == 5 && !p6.empty()) {
      world.send(0, 5, p6.data(), half_size_squared);
      std::cout << "[DEBUG] StrassenMultiply: Rank 5 sent p6" << std::endl;
    } else if (world.rank() == 6 && !p7.empty()) {
      world.send(0, 6, p7.data(), half_size_squared);
      std::cout << "[DEBUG] StrassenMultiply: Rank 6 sent p7" << std::endl;
    }
  }

  world.barrier();
  std::cout << "[DEBUG] StrassenMultiply: Rank = " << world.rank() << ", MPI Barrier passed" << std::endl;

  std::vector<double> result;
  if (world.rank() == 0) {
    std::vector<double> c11 =
        AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
    std::vector<double> c12 = AddMatrices(p3, p5, half_size);
    std::vector<double> c21 = AddMatrices(p2, p4, half_size);
    std::vector<double> c22 =
        AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

    result.resize(size * size);
    MergeMatrix(result, c11, 0, 0, size);
    MergeMatrix(result, c12, 0, half_size, size);
    MergeMatrix(result, c21, half_size, 0, size);
    MergeMatrix(result, c22, half_size, half_size, size);

    std::cout << "[DEBUG] StrassenMultiply: Rank 0, Result matrix computed" << std::endl;
  } else {
    result.resize(size * size, 0.0);
  }

  // Рассылаем результат всем процессам
  boost::mpi::broadcast(world, result.data(), size * size, 0);
  std::cout << "[DEBUG] StrassenMultiply: Rank = " << world.rank() << ", Result broadcasted" << std::endl;

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