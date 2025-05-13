#include "stl/nasedkin_e_strassen_algorithm/include/ops_stl.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <vector>

namespace nasedkin_e_strassen_algorithm_stl {

bool StrassenStl::PreProcessingImpl() {
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

bool StrassenStl::ValidationImpl() {
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

bool StrassenStl::RunImpl() {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_, world);
  return true;
}

bool StrassenStl::PostProcessingImpl() {
  if (original_size_ != matrix_size_) {
    output_matrix_ = TrimMatrixToOriginalSize(output_matrix_, original_size_, matrix_size_);
  }

  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(output_matrix_, out_ptr);
  return true;
}

std::vector<double> StrassenStl::AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::plus<>());
  return result;
}

std::vector<double> StrassenStl::SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b,
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

std::vector<double> StrassenStl::PadMatrixToPowerOfTwo(const std::vector<double>& matrix, int original_size) {
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

std::vector<double> StrassenStl::TrimMatrixToOriginalSize(const std::vector<double>& matrix, int original_size,
                                                          int padded_size) {
  std::vector<double> trimmed_matrix(original_size * original_size);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * padded_size, matrix.begin() + i * padded_size + original_size,
                      trimmed_matrix.begin() + i * original_size);
  }
  return trimmed_matrix;
}

std::vector<double> StrassenStl::StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b, int size,
                                                  boost::mpi::communicator& world) {
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

  std::vector<double> p1(half_size_squared);
  std::vector<double> p2(half_size_squared);
  std::vector<double> p3(half_size_squared);
  std::vector<double> p4(half_size_squared);
  std::vector<double> p5(half_size_squared);
  std::vector<double> p6(half_size_squared);
  std::vector<double> p7(half_size_squared);

  // Prepare submatrices for all processes
  std::vector<std::vector<double>> submatrices(6, std::vector<double>(half_size_squared));
  std::vector<std::vector<double>> submatrices_b(6, std::vector<double>(half_size_squared));

  if (world.rank() == 0) {
    submatrices[0] = AddMatrices(a21, a22, half_size);
    submatrices[1] = a11;
    submatrices[2] = a22;
    submatrices[3] = AddMatrices(a11, a12, half_size);
    submatrices[4] = SubtractMatrices(a21, a11, half_size);
    submatrices[5] = SubtractMatrices(a12, a22, half_size);

    submatrices_b[0] = b11;
    submatrices_b[1] = SubtractMatrices(b12, b22, half_size);
    submatrices_b[2] = SubtractMatrices(b21, b11, half_size);
    submatrices_b[3] = b22;
    submatrices_b[4] = AddMatrices(b11, b12, half_size);
    submatrices_b[5] = AddMatrices(b21, b22, half_size);
  }

  // Broadcast submatrices to all processes
  for (int i = 0; i < 6; ++i) {
    boost::mpi::broadcast(world, submatrices[i], 0);
    boost::mpi::broadcast(world, submatrices_b[i], 0);
  }

  if (world.rank() == 0) {
    // Master process computes p1
    p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, world);

    std::vector<std::vector<double>> results(7, std::vector<double>(half_size_squared));

    // Scatter tasks for p2 to p7
    if (world.size() > 1) {
      for (int i = 1; i <= 6; ++i) {
        world.send(i % world.size(), 0, i);
      }
      // Collect results from other processes
      for (int i = 1; i <= 6; ++i) {
        world.recv(boost::mpi::any_source, 1, results[i]);
      }
      p2 = results[1];
      p3 = results[2];
      p4 = results[3];
      p5 = results[4];
      p6 = results[5];
      p7 = results[6];
    } else {
      // Single process computes all
      p2 = StrassenMultiply(submatrices[0], submatrices_b[0], half_size, world);
      p3 = StrassenMultiply(submatrices[1], submatrices_b[1], half_size, world);
      p4 = StrassenMultiply(submatrices[2], submatrices_b[2], half_size, world);
      p5 = StrassenMultiply(submatrices[3], submatrices_b[3], half_size, world);
      p6 = StrassenMultiply(submatrices[4], submatrices_b[4], half_size, world);
      p7 = StrassenMultiply(submatrices[5], submatrices_b[5], half_size, world);
    }
  } else {
    // Worker processes
    int task_id;
    world.recv(0, 0, task_id);
    std::vector<double> result;
    switch (task_id) {
      case 1:
        result = StrassenMultiply(submatrices[0], submatrices_b[0], half_size, world);
        world.send(0, 1, result);
        break;
      case 2:
        result = StrassenMultiply(submatrices[1], submatrices_b[1], half_size, world);
        world.send(0, 1, result);
        break;
      case 3:
        result = StrassenMultiply(submatrices[2], submatrices_b[2], half_size, world);
        world.send(0, 1, result);
        break;
      case 4:
        result = StrassenMultiply(submatrices[3], submatrices_b[3], half_size, world);
        world.send(0, 1, result);
        break;
      case 5:
        result = StrassenMultiply(submatrices[4], submatrices_b[4], half_size, world);
        world.send(0, 1, result);
        break;
      case 6:
        result = StrassenMultiply(submatrices[5], submatrices_b[5], half_size, world);
        world.send(0, 1, result);
        break;
    }
  }

  // Synchronize all processes
  world.barrier();

  // Broadcast results to all processes
  boost::mpi::broadcast(world, p1, 0);
  boost::mpi::broadcast(world, p2, 0);
  boost::mpi::broadcast(world, p3, 0);
  boost::mpi::broadcast(world, p4, 0);
  boost::mpi::broadcast(world, p5, 0);
  boost::mpi::broadcast(world, p6, 0);
  boost::mpi::broadcast(world, p7, 0);

  std::vector<double> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  std::vector<double> c12 = AddMatrices(p3, p5, half_size);
  std::vector<double> c21 = AddMatrices(p2, p4, half_size);
  std::vector<double> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

  std::vector<double> result(size * size);
  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);

  return result;
}

void StrassenStl::SplitMatrix(const std::vector<double>& parent, std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(parent.begin() + (row_start + i) * parent_size + col_start,
                      parent.begin() + (row_start + i) * parent_size + col_start + child_size,
                      child.begin() + i * child_size);
  }
}

void StrassenStl::MergeMatrix(std::vector<double>& parent, const std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(child.begin() + i * child_size, child.begin() + (i + 1) * child_size,
                      parent.begin() + (row_start + i) * parent_size + col_start);
  }
}

}  // namespace nasedkin_e_strassen_algorithm_stl