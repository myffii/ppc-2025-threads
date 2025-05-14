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

StrassenAll::StrassenAll(ppc::core::TaskDataPtr task_data)
    : Task(std::move(task_data)), comm_(boost::mpi::communicator()) {
  std::cout << "[MPI Rank " << comm_.rank() << "] Constructor called\n";
}

bool StrassenAll::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* in_ptr_b = reinterpret_cast<double*>(task_data->inputs[1]);

  matrix_size_ = static_cast<int>(std::sqrt(input_size));
  input_matrix_a_.resize(matrix_size_ * matrix_size_);
  input_matrix_b_.resize(matrix_size_ * matrix_size_);

  std::cout << "[MPI Rank " << comm_.rank() << "] PreProcessing: Copying input matrices of size " << matrix_size_ << "x"
            << matrix_size_ << "\n";

  std::ranges::copy(in_ptr_a, in_ptr_a + input_size, input_matrix_a_.begin());
  std::ranges::copy(in_ptr_b, in_ptr_b + input_size, input_matrix_b_.begin());

  if ((matrix_size_ & (matrix_size_ - 1)) != 0) {
    original_size_ = matrix_size_;
    input_matrix_a_ = PadMatrixToPowerOfTwo(input_matrix_a_, matrix_size_, comm_);
    input_matrix_b_ = PadMatrixToPowerOfTwo(input_matrix_b_, matrix_size_, comm_);
    matrix_size_ = static_cast<int>(std::sqrt(input_matrix_a_.size()));
    std::cout << "[MPI Rank " << comm_.rank() << "] PreProcessing: Padded matrices to size " << matrix_size_ << "x"
              << matrix_size_ << "\n";
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
    std::cout << "[MPI Rank " << comm_.rank() << "] Validation failed: Zero input/output size\n";
    return false;
  }

  int size_a = static_cast<int>(std::sqrt(input_size_a));
  int size_b = static_cast<int>(std::sqrt(input_size_b));
  int size_output = static_cast<int>(std::sqrt(output_size));

  bool valid = (size_a == size_b) && (size_a == size_output);
  std::cout << "[MPI Rank " << comm_.rank() << "] Validation: " << (valid ? "Passed" : "Failed")
            << " (size_a=" << size_a << ", size_b=" << size_b << ", size_output=" << size_output << ")\n";
  return valid;
}

bool StrassenAll::RunImpl() {
  std::cout << "[MPI Rank " << comm_.rank() << "] Running StrassenMultiply with matrix size " << matrix_size_ << "x"
            << matrix_size_ << "\n";
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_, comm_);
  return true;
}

bool StrassenAll::PostProcessingImpl() {
  if (original_size_ != matrix_size_) {
    std::cout << "[MPI Rank " << comm_.rank() << "] PostProcessing: Trimming matrix from " << matrix_size_ << "x"
              << matrix_size_ << " to " << original_size_ << "x" << original_size_ << "\n";
    output_matrix_ = TrimMatrixToOriginalSize(output_matrix_, original_size_, matrix_size_, comm_);
  }

  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(output_matrix_, out_ptr);
  std::cout << "[MPI Rank " << comm_.rank() << "] PostProcessing: Copied output matrix\n";
  return true;
}

std::vector<double> StrassenAll::AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::vector<double> result(size * size);
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads;
  const int chunk_size = (size * size + num_threads - 1) / num_threads;

  std::cout << "[MPI Rank " << boost::mpi::communicator().rank() << "] AddMatrices: Using " << num_threads
            << " threads for size " << size << "x" << size << "\n";

  for (int t = 0; t < num_threads; ++t) {
    int start = t * chunk_size;
    int end = std::min(start + chunk_size, size * size);
    if (start < end) {
      threads.emplace_back([&, start, end, t]() {
        std::cout << "[MPI Rank " << boost::mpi::communicator().rank() << ", Thread " << t
                  << "] Adding matrices from index " << start << " to " << end << "\n";
        for (int i = start; i < end; ++i) {
          result[i] = a[i] + b[i];
        }
      });
    }
  }

  for (auto& thread : threads) {
    thread.join();
  }
  return result;
}

std::vector<double> StrassenAll::SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b,
                                                  int size) {
  std::vector<double> result(size * size);
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads;
  const int chunk_size = (size * size + num_threads - 1) / num_threads;

  std::cout << "[MPI Rank " << boost::mpi::communicator().rank() << "] SubtractMatrices: Using " << num_threads
            << " threads for size " << size << "x" << size << "\n";

  for (int t = 0; t < num_threads; ++t) {
    int start = t * chunk_size;
    int end = std::min(start + chunk_size, size * size);
    if (start < end) {
      threads.emplace_back([&, start, end, t]() {
        std::cout << "[MPI Rank " << boost::mpi::communicator().rank() << ", Thread " << t
                  << "] Subtracting matrices from index " << start << " to " << end << "\n";
        for (int i = start; i < end; ++i) {
          result[i] = a[i] - b[i];
        }
      });
    }
  }

  for (auto& thread : threads) {
    thread.join();
  }
  return result;
}

std::vector<double> StandardMultiply(const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::vector<double> result(size * size, 0.0);
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads;
  const int chunk_size = (size + num_threads - 1) / num_threads;

  std::cout << "[MPI Rank " << boost::mpi::communicator().rank() << "] StandardMultiply: Using " << num_threads
            << " threads for size " << size << "x" << size << "\n";

  for (int t = 0; t < num_threads; ++t) {
    int start_i = t * chunk_size;
    int end_i = std::min(start_i + chunk_size, size);
    if (start_i < end_i) {
      threads.emplace_back([&, start_i, end_i, t]() {
        std::cout << "[MPI Rank " << boost::mpi::communicator().rank() << ", Thread " << t
                  << "] StandardMultiply from row " << start_i << " to " << end_i << "\n";
        for (int i = start_i; i < end_i; ++i) {
          for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
              result[(i * size) + j] += a[(i * size) + k] * b[(k * size) + j];
            }
          }
        }
      });
    }
  }

  for (auto& thread : threads) {
    thread.join();
  }
  return result;
}

std::vector<double> StrassenAll::PadMatrixToPowerOfTwo(const std::vector<double>& matrix, int original_size,
                                                       boost::mpi::communicator comm) {
  int new_size = 1;
  while (new_size < original_size) {
    new_size *= 2;
  }

  std::vector<double> padded_matrix(new_size * new_size, 0);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * original_size, matrix.begin() + (i + 1) * original_size,
                      padded_matrix.begin() + i * new_size);
  }
  std::cout << "[MPI Rank " << comm.rank() << "] PadMatrixToPowerOfTwo: Padded from " << original_size << "x"
            << original_size << " to " << new_size << "x" << new_size << "\n";
  return padded_matrix;
}

std::vector<double> StrassenAll::TrimMatrixToOriginalSize(const std::vector<double>& matrix, int original_size,
                                                          int padded_size, boost::mpi::communicator comm) {
  std::vector<double> trimmed_matrix(original_size * original_size);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * padded_size, matrix.begin() + i * padded_size + original_size,
                      trimmed_matrix.begin() + i * original_size);
  }
  std::cout << "[MPI Rank " << comm.rank() << "] TrimMatrixToOriginalSize: Trimmed from " << padded_size << "x"
            << padded_size << " to " << original_size << "x" << original_size << "\n";
  return trimmed_matrix;
}

std::vector<double> StrassenAll::StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b, int size,
                                                  boost::mpi::communicator comm) {
  if (size <= 32) {
    std::cout << "[MPI Rank " << comm.rank() << "] StrassenMultiply: Switching to StandardMultiply for size " << size
              << "x" << size << "\n";
    return StandardMultiply(a, b, size);
  }

  int half_size = size / 2;
  int half_size_squared = half_size * half_size;

  std::cout << "[MPI Rank " << comm.rank() << "] StrassenMultiply: Processing size " << size << "x" << size
            << ", half_size=" << half_size << "\n";

  std::vector<double> a11(half_size_squared);
  std::vector<double> a12(half_size_squared);
  std::vector<double> a21(half_size_squared);
  std::vector<double> a22(half_size_squared);
  std::vector<double> b11(half_size_squared);
  std::vector<double> b12(half_size_squared);
  std::vector<double> b21(half_size_squared);
  std::vector<double> b22(half_size_squared);

  std::cout << "[MPI Rank " << comm.rank() << "] Splitting matrices\n";
  SplitMatrix(a, a11, 0, 0, size, comm);
  SplitMatrix(a, a12, 0, half_size, size, comm);
  SplitMatrix(a, a21, half_size, 0, size, comm);
  SplitMatrix(a, a22, half_size, half_size, size, comm);
  SplitMatrix(b, b11, 0, 0, size, comm);
  SplitMatrix(b, b12, 0, half_size, size, comm);
  SplitMatrix(b, b21, half_size, 0, size, comm);
  SplitMatrix(b, b22, half_size, half_size, size, comm);

  std::vector<double> p1(half_size_squared);
  std::vector<double> p2(half_size_squared);
  std::vector<double> p3(half_size_squared);
  std::vector<double> p4(half_size_squared);
  std::vector<double> p5(half_size_squared);
  std::vector<double> p6(half_size_squared);
  std::vector<double> p7(half_size_squared);

  const int rank = comm.rank();
  const int world_size = comm.size();
  const int num_tasks = 7;
  std::vector<std::vector<double>> local_results;

  std::cout << "[MPI Rank " << rank << "] StrassenMultiply: World size=" << world_size << ", computing tasks\n";

  for (int task_id = rank; task_id < num_tasks; task_id += world_size) {
    std::cout << "[MPI Rank " << rank << "] Computing task p" << (task_id + 1) << "\n";
    std::vector<double> result;
    switch (task_id) {
      case 0:
        result = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, comm);
        break;
      case 1:
        result = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, comm);
        break;
      case 2:
        result = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, comm);
        break;
      case 3:
        result = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, comm);
        break;
      case 4:
        result = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, comm);
        break;
      case 5:
        result =
            StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size, comm);
        break;
      case 6:
        result =
            StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size, comm);
        break;
    }
    local_results.push_back(result);
  }

  std::cout << "[MPI Rank " << rank << "] Broadcasting results\n";
  for (int task_id = 0; task_id < num_tasks; ++task_id) {
    int owner_rank = task_id % world_size;
    std::vector<double> result(half_size_squared);
    if (rank == owner_rank) {
      result = local_results[task_id / world_size];
    }
    boost::mpi::broadcast(comm, result, owner_rank);
    switch (task_id) {
      case 0:
        p1 = result;
        break;
      case 1:
        p2 = result;
        break;
      case 2:
        p3 = result;
        break;
      case 3:
        p4 = result;
        break;
      case 4:
        p5 = result;
        break;
      case 5:
        p6 = result;
        break;
      case 6:
        p7 = result;
        break;
    }
  }

  std::cout << "[MPI Rank " << rank << "] Computing final matrices\n";
  std::vector<double> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  std::vector<double> c12 = AddMatrices(p3, p5, half_size);
  std::vector<double> c21 = AddMatrices(p2, p4, half_size);
  std::vector<double> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

  std::vector<double> result(size * size);
  MergeMatrix(result, c11, 0, 0, size, comm);
  MergeMatrix(result, c12, 0, half_size, size, comm);
  MergeMatrix(result, c21, half_size, 0, size, comm);
  MergeMatrix(result, c22, half_size, half_size, size, comm);

  std::cout << "[MPI Rank " << rank << "] StrassenMultiply completed for size " << size << "x" << size << "\n";
  return result;
}

void StrassenAll::SplitMatrix(const std::vector<double>& parent, std::vector<double>& child, int row_start,
                              int col_start, int parent_size, boost::mpi::communicator comm) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(parent.begin() + (row_start + i) * parent_size + col_start,
                      parent.begin() + (row_start + i) * parent_size + col_start + child_size,
                      child.begin() + i * child_size);
  }
  std::cout << "[MPI Rank " << comm.rank() << "] SplitMatrix: Copied from (" << row_start << "," << col_start
            << ") to child of size " << child_size << "x" << child_size << "\n";
}

void StrassenAll::MergeMatrix(std::vector<double>& parent, const std::vector<double>& child, int row_start,
                              int col_start, int parent_size, boost::mpi::communicator comm) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(child.begin() + i * child_size, child.begin() + (i + 1) * child_size,
                      parent.begin() + (row_start + i) * parent_size + col_start);
  }
  std::cout << "[MPI Rank " << comm.rank() << "] MergeMatrix: Copied to (" << row_start << "," << col_start
            << ") in parent of size " << parent_size << "x" << parent_size << "\n";
}

}  // namespace nasedkin_e_strassen_algorithm_all