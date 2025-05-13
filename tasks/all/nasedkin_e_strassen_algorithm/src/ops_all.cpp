#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace nasedkin_e_strassen_algorithm_all {

bool StrassenAll::PreProcessingImpl() {
  boost::mpi::communicator StrassenAll::mpi_comm_;  
  int StrassenAll::mpi_world_size_ = 0;
  int StrassenAll::mpi_rank_ = 0;

  if (mpi_rank_ == 0) {
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

  mpi_comm_.broadcast(matrix_size_, 0);
  mpi_comm_.broadcast(original_size_, 0);

  if (mpi_rank_ != 0) {
    input_matrix_a_.resize(matrix_size_ * matrix_size_);
    input_matrix_b_.resize(matrix_size_ * matrix_size_);
    output_matrix_.resize(matrix_size_ * matrix_size_, 0.0);
  }

  mpi_comm_.broadcast(input_matrix_a_, 0);
  mpi_comm_.broadcast(input_matrix_b_, 0);

  return true;
}

bool StrassenAll::ValidationImpl() {
  if (mpi_rank_ == 0) {
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
  if (mpi_rank_ == 0) {
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
  p1.resize(half_size_squared);
  p2.resize(half_size_squared);
  p3.resize(half_size_squared);
  p4.resize(half_size_squared);
  p5.resize(half_size_squared);
  p6.resize(half_size_squared);
  p7.resize(half_size_squared);

  int task_assigned = mpi_rank_ % 7;

  std::vector<double> s1, s2, s3, s4, s5, s6, s7, s8, s9, s10;

  switch (task_assigned) {
    case 0:
      s1 = AddMatrices(a11, a22, half_size);
      s2 = AddMatrices(b11, b22, half_size);
      p1 = StrassenMultiply(s1, s2, half_size, num_threads);
      break;
    case 1:
      s3 = AddMatrices(a21, a22, half_size);
      p2 = StrassenMultiply(s3, b11, half_size, num_threads);
      break;
    case 2:
      s4 = SubtractMatrices(b12, b22, half_size);
      p3 = StrassenMultiply(a11, s4, half_size, num_threads);
      break;
    case 3:
      s5 = SubtractMatrices(b21, b11, half_size);
      p4 = StrassenMultiply(a22, s5, half_size, num_threads);
      break;
    case 4:
      s6 = AddMatrices(a11, a12, half_size);
      p5 = StrassenMultiply(s6, b22, half_size, num_threads);
      break;
    case 5:
      s7 = SubtractMatrices(a21, a11, half_size);
      s8 = AddMatrices(b11, b12, half_size);
      p6 = StrassenMultiply(s7, s8, half_size, num_threads);
      break;
    case 6:
      s9 = SubtractMatrices(a12, a22, half_size);
      s10 = AddMatrices(b21, b22, half_size);
      p7 = StrassenMultiply(s9, s10, half_size, num_threads);
      break;
  }

  mpi_comm_.all_gather(p1, p1);
  mpi_comm_.all_gather(p2, p2);
  mpi_comm_.all_gather(p3, p3);
  mpi_comm_.all_gather(p4, p4);
  mpi_comm_.all_gather(p5, p5);
  mpi_comm_.all_gather(p6, p6);
  mpi_comm_.all_gather(p7, p7);

  std::vector<double> c11, c12, c21, c22;
  c11.resize(half_size_squared);
  c12.resize(half_size_squared);
  c21.resize(half_size_squared);
  c22.resize(half_size_squared);

  std::vector<std::function<void()>> tasks;
  tasks.reserve(4);
  tasks.emplace_back([&]() {
    auto temp1 = AddMatrices(p1, p4, half_size);
    auto temp2 = SubtractMatrices(temp1, p5, half_size);
    c11 = AddMatrices(temp2, p7, half_size);
  });
  tasks.emplace_back([&]() { c12 = AddMatrices(p3, p5, half_size); });
  tasks.emplace_back([&]() { c21 = AddMatrices(p2, p4, half_size); });
  tasks.emplace_back([&]() {
    auto temp1 = AddMatrices(p1, p3, half_size);
    auto temp2 = SubtractMatrices(temp1, p2, half_size);
    c22 = AddMatrices(temp2, p6, half_size);
  });

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

  std::vector<double> result(size * size);
  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);

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