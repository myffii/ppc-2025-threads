#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "boost/mpi/collectives/reduce.hpp"
#include "boost/mpi/communicator.hpp"
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
  boost::mpi::communicator world;
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_, num_threads, world);
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

void StrassenAll::AddMatrices(std::vector<double>& result, const std::vector<double>& a, const std::vector<double>& b,
                              int size) {
  result.resize(size * size);
  std::ranges::transform(a, b, result.begin(), std::plus<>());
}

void StrassenAll::SubtractMatrices(std::vector<double>& result, const std::vector<double>& a,
                                   const std::vector<double>& b, int size) {
  result.resize(size * size);
  std::ranges::transform(a, b, result.begin(), std::minus<>());
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
                                                  int num_threads, boost::mpi::communicator& world) {
  if (size <= 64) {
    return StandardMultiply(a, b, size);
  }

  int rank = world.rank();
  int num_processes = world.size();
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

  if (rank == 0) {
    SplitMatrix(a, a11, 0, 0, size);
    SplitMatrix(a, a12, 0, half_size, size);
    SplitMatrix(a, a21, half_size, 0, size);
    SplitMatrix(a, a22, half_size, half_size, size);
    SplitMatrix(b, b11, 0, 0, size);
    SplitMatrix(b, b12, 0, half_size, size);
    SplitMatrix(b, b21, half_size, 0, size);
    SplitMatrix(b, b22, half_size, half_size, size);
  }

  boost::mpi::broadcast(world, a11, 0);
  boost::mpi::broadcast(world, a12, 0);
  boost::mpi::broadcast(world, a21, 0);
  boost::mpi::broadcast(world, a22, 0);
  boost::mpi::broadcast(world, b11, 0);
  boost::mpi::broadcast(world, b12, 0);
  boost::mpi::broadcast(world, b21, 0);
  boost::mpi::broadcast(world, b22, 0);

  std::vector<double> p1(half_size_squared, 0.0);
  std::vector<double> p2(half_size_squared, 0.0);
  std::vector<double> p3(half_size_squared, 0.0);
  std::vector<double> p4(half_size_squared, 0.0);
  std::vector<double> p5(half_size_squared, 0.0);
  std::vector<double> p6(half_size_squared, 0.0);
  std::vector<double> p7(half_size_squared, 0.0);

  std::vector<std::function<void()>> tasks;
  for (int i = 0; i < 7; ++i) {
    if (rank % num_processes == i % num_processes) {
      switch (i) {
        case 0: {
          std::vector<double> temp_a(half_size_squared), temp_b(half_size_squared);
          tasks.emplace_back([&, temp_a = std::move(temp_a), temp_b = std::move(temp_b)]() mutable {
            AddMatrices(temp_a, a11, a22, half_size);
            AddMatrices(temp_b, b11, b22, half_size);
            p1 = StrassenMultiply(temp_a, temp_b, half_size, num_threads, world);
          });
          break;
        }
        case 1: {
          std::vector<double> temp_a(half_size_squared);
          tasks.emplace_back([&, temp_a = std::move(temp_a)]() mutable {
            AddMatrices(temp_a, a21, a22, half_size);
            p2 = StrassenMultiply(temp_a, b11, half_size, num_threads, world);
          });
          break;
        }
        case 2: {
          std::vector<double> temp_b(half_size_squared);
          tasks.emplace_back([&, temp_b = std::move(temp_b)]() mutable {
            SubtractMatrices(temp_b, b12, b22, half_size);
            p3 = StrassenMultiply(a11, temp_b, half_size, num_threads, world);
          });
          break;
        }
        case 3: {
          std::vector<double> temp_b(half_size_squared);
          tasks.emplace_back([&, temp_b = std::move(temp_b)]() mutable {
            SubtractMatrices(temp_b, b21, b11, half_size);
            p4 = StrassenMultiply(a22, temp_b, half_size, num_threads, world);
          });
          break;
        }
        case 4: {
          std::vector<double> temp_a(half_size_squared);
          tasks.emplace_back([&, temp_a = std::move(temp_a)]() mutable {
            AddMatrices(temp_a, a11, a12, half_size);
            p5 = StrassenMultiply(temp_a, b22, half_size, num_threads, world);
          });
          break;
        }
        case 5: {
          std::vector<double> temp_a(half_size_squared), temp_b(half_size_squared);
          tasks.emplace_back([&, temp_a = std::move(temp_a), temp_b = std::move(temp_b)]() mutable {
            SubtractMatrices(temp_a, a21, a11, half_size);
            AddMatrices(temp_b, b11, b12, half_size);
            p6 = StrassenMultiply(temp_a, temp_b, half_size, num_threads, world);
          });
          break;
        }
        case 6: {
          std::vector<double> temp_a(half_size_squared), tempТаблица б(half_size_squared);
          tasks.emplace_back([&, temp_a = std::move(temp_a), temp_b = std::move(temp_b)]() mutable {
            SubtractMatrices(temp_a, a12, a22, half_size);
            AddMatrices(temp_b, b21, b22, half_size);
            p7 = StrassenMultiply(temp_a, temp_b, half_size, num_threads, world);
          });
          break;
        }
      }
    }
  }

  std::vector<std::thread> threads;
  threads.reserve(std::min(num_threads, static_cast<int>(tasks.size())));
  size_t task_index = 0;

  for (int i = 0; i < std::min(num_threads, static_cast<int>(tasks.size())); ++i) {
    if (task_index < tasks.size()) {
      threads.emplace_back(tasks[task_index]);
      ++task_index;
    }
  }

  while (task_index < tasks.size()) {
    tasks[task_index]();
    ++task_index;
  }

  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  world.barrier();

  std::vector<double> p1_out(half_size_squared, 0.0);
  std::vector<double> p2_out(half_size_squared, 0.0);
  std::vector<double> p3_out(half_size_squared, 0.0);
  std::vector<double> p4_out(half_size_squared, 0.0);
  std::vector<double> p5_out(half_size_squared, 0.0);
  std::vector<double> p6_out(half_size_squared, 0.0);
  std::vector<double> p7_out(half_size_squared, 0.0);

  boost::mpi::reduce(world, p1.data(), static_cast<int>(p1.size()), p1_out.data(), std::plus<>{}, 0);
  boost::mpi::reduce(world, p2.data(), static_cast<int>(p2.size()), p2_out.data(), std::plus<>{}, 0);
  boost::mpi::reduce(world, p3.data(), static_cast<int>(p3.size()), p3_out.data(), std::plus<>{}, 0);
  boost::mpi::reduce(world, p4.data(), static_cast<int>(p4.size()), p4_out.data(), std::plus<>{}, 0);
  boost::mpi::reduce(world, p5.data(), static_cast<int>(p5.size()), p5_out.data(), std::plus<>{}, 0);
  boost::mpi::reduce(world, p6.data(), static_cast<int>(p6.size()), p6_out.data(), std::plus<>{}, 0);
  boost::mpi::reduce(world, p7.data(), static_cast<int>(p7.size()), p7_out.data(), std::plus<>{}, 0);

  std::vector<double> result(size * size, 0.0);
  if (rank == 0) {
    std::vector<double> c11(half_size_squared);
    std::vector<double> c12(half_size_squared);
    std::vector<double> c21(half_size_squared);
    std::vector<double> c22(half_size_squared);

    AddMatrices(c11, p1_out, p4_out, half_size);
    SubtractMatrices(c11, c11, p5_out, half_size);
    AddMatrices(c11, c11, p7_out, half_size);

    AddMatrices(c12, p3_out, p5_out, half_size);

    AddMatrices(c21, p2_out, p4_out, half_size);

    AddMatrices(c22, p1_out, p3_out, half_size);
    SubtractMatrices(c22, c22, p2_out, half_size);
    AddMatrices(c22, c22, p6_out, half_size);

    MergeMatrix(result, c11, 0, 0, size);
    MergeMatrix(result, c12, 0, half_size, size);
    MergeMatrix(result, c21, half_size, 0, size);
    MergeMatrix(result, c22, half_size, half_size, size);
  }

  boost::mpi::broadcast(world, result, 0);
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