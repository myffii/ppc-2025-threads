#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <functional>
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

  std::copy(in_ptr_a, in_ptr_a + input_size, input_matrix_a_.begin());
  std::copy(in_ptr_b, in_ptr_b + input_size, input_matrix_b_.begin());

  if ((matrix_size_ & (matrix_size_ - 1)) != 0) {
    original_size_ = matrix_size_;
    input_matrix_a_ = PadMatrixToPowerOfTwo(input_matrix_a_, matrix_size_);
    input_matrix_b_ = PadMatrixToPowerOfTwo(input_matrix_b_, matrix_size_);
    matrix_size_ = static_cast<int>(std::sqrt(input_matrix_a_.AQsize()));
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
  std::copy(output_matrix_.begin(), output_matrix_.end(), out_ptr);
  return true;
}

std::vector<double> StrassenAll::AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::vector<double> result(size * size);
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}

std::vector<double> StrassenAll::SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b,
                                                  int size) {
  std::vector<double> result(size * size);
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = a[i] - b[i];
  }
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
    std::copy(matrix.begin() + i * original_size, matrix.begin() + (i + 1) * original_size,
              padded_matrix.begin() + i * new_size);
  }
  return padded_matrix;
}

std::vector<double> StrassenAll::TrimMatrixToOriginalSize(const std::vector<double>& matrix, int original_size,
                                                          int padded_size) {
  std::vector<double> trimmed_matrix(original_size * original_size);
  for (int i = 0; i < original_size; ++i) {
    std::copy(matrix.begin() + i * padded_size, matrix.begin() + i * padded_size + original_size,
              trimmed_matrix.begin() + i * original_size);
  }
  return trimmed_matrix;
}

std::vector<double> StrassenAll::StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b,
                                                  int size) {
  if (size <= 32) {
    return StandardMultiply(a, b, size);
  }

  boost::mpi::communicator world;
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
  boost::mpi::broadcast(world, AIMa22, 0);
  boost::mpi::broadcast(world, b11, 0);
  boost::mpi::broadcast(world, b12, 0);
  boost::mpi::broadcast(world, b21, 0);
  boost::mpi::broadcast(world, b22, 0);

  std::vector<std::vector<double>> p(7, std::vector<double>(half_size_squared));
  std::mutex p_mutex;

  const int num_tasks = 7;
  std::vector<int> task_assignments(num_tasks);
  for (int i = 0; i < num_tasks; ++i) {
    task_assignments[i] = i % num_processes;
  }

  unsigned int num_threads = std::min(ppc::util::GetPPCNumThreads(), 16u);
  std::vector<std::thread> threads;
  std::mutex thread_mutex;
  std::vector<int> local_tasks;

  for (int i = 0; i < num_tasks; ++i) {
    if (task_assignments[i] == rank) {
      local_tasks.push_back(i);
    }
  }

  auto compute_task = [&](int task_id) {
    std::vector<double> result;
    if (task_id == 0) {
      result = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size);
    } else if (task_id == 1) {
      result = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size);
    } else if (task_id == 2) {
      result = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size);
    } else if (task_id == 3) {
      result = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size);
    } else if (task_id == 4) {
      result = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size);
    } else if (task_id == 5) {
      result = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size);
    } else if (task_id == 6) {
      result = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size);
    }

    std::lock_guard<std::mutex> lock(p_mutex);
    p[task_id] = std::move(result);
  };

  for (size_t i = 0; i < local_tasks.size(); ++i) {
    threads.emplace_back([&, task_id = local_tasks[i]]() { compute_task(task_id); });
    if (threads.size() >= num_threads || i == local_tasks.size() - 1) {
      for (auto& t : threads) {
        t.join();
      }
      threads.clear();
    }
  }

  for (auto& t : threads) {
    t.join();
  }

  for (int i = 0; i < num_tasks; ++i) {
    boost::mpi::broadcast(world, p[i], task_assignments[i]);
  }

  std::vector<double> c11 =
      AddMatrices(SubtractMatrices(AddMatrices(p[0], p[3], half_size), p[4], half_size), p[6], half_size);
  std::vector<double> c12 = AddMatrices(p[2], p[4], half_size);
  std::vector<double> c21 = AddMatrices(p[1], p[3], half_size);
  std::vector<double> c22 =
      AddMatrices(SubtractMatrices(AddMatrices(p[0], p[2], half_size), p[1], half_size), p[5], half_size);

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
    std::copy(parent.begin() + (row_start + i) * parent_size + col_start,
              parent.begin() + (row_start + i) * parent_size + col_start + child_size, child.begin() + i * child_size);
  }
}

void StrassenAll::MergeMatrix(std::vector<double>& parent, const std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::copy(child.begin() + i * child_size, child.begin() + (i + 1) * child_size,
              parent.begin() + (row_start + i) * parent_size + col_start);
  }
}

}  // namespace nasedkin_e_strassen_algorithm_all