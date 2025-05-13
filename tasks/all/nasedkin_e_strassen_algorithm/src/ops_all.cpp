#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
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

  std::cout << "Process " << boost::mpi::communicator().rank()
            << ": PreProcessing started, matrix size = " << matrix_size_ << std::endl;

  std::ranges::copy(in_ptr_a, in_ptr_a + input_size, input_matrix_a_.begin());
  std::ranges::copy(in_ptr_b, in_ptr_b + input_size, input_matrix_b_.begin());

  if ((matrix_size_ & (matrix_size_ - 1)) != 0) {
    original_size_ = matrix_size_;
    input_matrix_a_ = PadMatrixToPowerOfTwo(input_matrix_a_, matrix_size_);
    input_matrix_b_ = PadMatrixToPowerOfTwo(input_matrix_b_, matrix_size_);
    matrix_size_ = static_cast<int>(std::sqrt(input_matrix_a_.size()));
    std::cout << "Process " << boost::mpi::communicator().rank() << ": Padded matrix to size = " << matrix_size_
              << std::endl;
  } else {
    original_size_ = matrix_size_;
  }

  output_matrix_.resize(matrix_size_ * matrix_size_, 0.0);
  std::cout << "Process " << boost::mpi::communicator().rank() << ": PreProcessing completed" << std::endl;
  return true;
}

bool StrassenAll::ValidationImpl() {
  unsigned int input_size_a = task_data->inputs_count[0];
  unsigned int input_size_b = task_data->inputs_count[1];
  unsigned int output_size = task_data->outputs_count[0];

  if (input_size_a == 0 || input_size_b == 0 || output_size == 0) {
    std::cout << "Process " << boost::mpi::communicator().rank() << ": Validation failed: Zero size detected"
              << std::endl;
    return false;
  }

  int size_a = static_cast<int>(std::sqrt(input_size_a));
  int size_b = static_cast<int>(std::sqrt(input_size_b));
  int size_output = static_cast<int>(std::sqrt(output_size));

  bool valid = (size_a == size_b) && (size_a == size_output);
  std::cout << "Process " << boost::mpi::communicator().rank() << ": Validation " << (valid ? "passed" : "failed")
            << ", sizes: a=" << size_a << ", b=" << size_b << ", out=" << size_output << std::endl;
  return valid;
}

bool StrassenAll::RunImpl() {
  boost::mpi::communicator comm;
  int num_threads = std::min(16, ppc::util::GetPPCNumThreads());
  std::cout << "Process " << comm.rank() << ": RunImpl started with " << num_threads << " threads" << std::endl;
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_, num_threads, comm);
  std::cout << "Process " << comm.rank() << ": RunImpl completed" << std::endl;
  return true;
}

bool StrassenAll::PostProcessingImpl() {
  if (original_size_ != matrix_size_) {
    output_matrix_ = TrimMatrixToOriginalSize(output_matrix_, original_size_, matrix_size_);
    std::cout << "Process " << boost::mpi::communicator().rank()
              << ": Trimmed matrix to original size = " << original_size_ << std::endl;
  }

  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(output_matrix_, out_ptr);
  std::cout << "Process " << boost::mpi::communicator().rank() << ": PostProcessing completed" << std::endl;
  return true;
}

std::vector<double> StrassenAll::AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::plus<>());
  std::cout << "Process " << boost::mpi::communicator().rank() << ": AddMatrices completed for size = " << size
            << std::endl;
  return result;
}

std::vector<double> StrassenAll::SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b,
                                                  int size) {
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::minus<>());
  std::cout << "Process " << boost::mpi::communicator().rank() << ": SubtractMatrices completed for size = " << size
            << std::endl;
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
  std::cout << "Process " << boost::mpi::communicator().rank() << ": StandardMultiply completed for size = " << size
            << std::endl;
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
  std::cout << "Process " << boost::mpi::communicator().rank()
            << ": PadMatrixToPowerOfTwo completed, new size = " << new_size << std::endl;
  return padded_matrix;
}

std::vector<double> StrassenAll::TrimMatrixToOriginalSize(const std::vector<double>& matrix, int original_size,
                                                          int padded_size) {
  std::vector<double> trimmed_matrix(original_size * original_size);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * padded_size, matrix.begin() + i * padded_size + original_size,
                      trimmed_matrix.begin() + i * original_size);
  }
  std::cout << "Process " << boost::mpi::communicator().rank()
            << ": TrimMatrixToOriginalSize completed, original size = " << original_size << std::endl;
  return trimmed_matrix;
}

std::vector<double> StrassenAll::StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b, int size,
                                                  int num_threads, boost::mpi::communicator& comm) {
  if (size <= 32) {
    std::cout << "Process " << comm.rank() << ": Using StandardMultiply for small matrix size = " << size << std::endl;
    return StandardMultiply(a, b, size);
  }

  std::cout << "Process " << comm.rank() << ": StrassenMultiply started, size = " << size
            << ", threads = " << num_threads << std::endl;

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

  std::cout << "Process " << comm.rank() << ": Splitting matrices" << std::endl;
  SplitMatrix(a, a11, 0, 0, size);
  SplitMatrix(a, a12, 0, half_size, size);
  SplitMatrix(a, a21, half_size, 0, size);
  SplitMatrix(a, a22, half_size, half_size, size);
  SplitMatrix(b, b11, 0, 0, size);
  SplitMatrix(b, b12, 0, half_size, size);
  SplitMatrix(b, b21, half_size, 0, size);
  SplitMatrix(b, b22, half_size, half_size, size);
  std::cout << "Process " << comm.rank() << ": Matrices split" << std::endl;

  std::vector<double> p1(half_size_squared);
  std::vector<double> p2(half_size_squared);
  std::vector<double> p3(half_size_squared);
  std::vector<double> p4(half_size_squared);
  std::vector<double> p5(half_size_squared);
  std::vector<double> p6(half_size_squared);
  std::vector<double> p7(half_size_squared);

  std::vector<std::function<void()>> tasks;
  tasks.reserve(7);
  tasks.emplace_back([&]() {
    std::cout << "Process " << comm.rank() << ": Computing p1" << std::endl;
    p1 = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size, num_threads,
                          comm);
    std::cout << "Process " << comm.rank() << ": p1 computed" << std::endl;
  });
  tasks.emplace_back([&]() {
    std::cout << "Process " << comm.rank() << ": Computing p2" << std::endl;
    p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads, comm);
    std::cout << "Process " << comm.rank() << ": p2 computed" << std::endl;
  });
  tasks.emplace_back([&]() {
    std::cout << "Process " << comm.rank() << ": Computing p3" << std::endl;
    p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads, comm);
    std::cout << "Process " << comm.rank() << ": p3 computed" << std::endl;
  });
  tasks.emplace_back([&]() {
    std::cout << "Process " << comm.rank() << ": Computing p4" << std::endl;
    p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads, comm);
    std::cout << "Process " << comm.rank() << ": p4 computed" << std::endl;
  });
  tasks.emplace_back([&]() {
    std::cout << "Process " << comm.rank() << ": Computing p5" << std::endl;
    p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads, comm);
    std::cout << "Process " << comm.rank() << ": p5 computed" << std::endl;
  });
  tasks.emplace_back([&]() {
    std::cout << "Process " << comm.rank() << ": Computing p6" << std::endl;
    p6 = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                          num_threads, comm);
    std::cout << "Process " << comm.rank() << ": p6 computed" << std::endl;
  });
  tasks.emplace_back([&]() {
    std::cout << "Process " << comm.rank() << ": Computing p7" << std::endl;
    p7 = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                          num_threads, comm);
    std::cout << "Process " << comm.rank() << ": p7 computed" << std::endl;
  });

  // Распределяем задачи между MPI-процессами
  int rank = comm.rank();
  int size_comm = comm.size();
  std::vector<std::function<void()>> local_tasks;
  for (size_t i = rank; i < tasks.size(); i += size_comm) {
    local_tasks.push_back(tasks[i]);
    std::cout << "Process " << rank << ": Assigned task " << i << std::endl;
  }

  // Выполняем локальные задачи в потоках
  std::vector<std::thread> threads;
  threads.reserve(std::min(num_threads, static_cast<int>(local_tasks.size())));
  size_t task_index = 0;

  std::cout << "Process " << rank << ": Starting " << local_tasks.size() << " tasks with " << num_threads << " threads"
            << std::endl;
  for (int i = 0; i < std::min(num_threads, static_cast<int>(local_tasks.size())); ++i) {
    if (task_index < local_tasks.size()) {
      threads.emplace_back(local_tasks[task_index]);
      std::cout << "Process " << rank << ": Thread " << i << " started for task " << task_index << std::endl;
      ++task_index;
    }
  }

  while (task_index < local_tasks.size()) {
    local_tasks[task_index]();
    std::cout << "Process " << rank << ": Executed task " << task_index << " sequentially" << std::endl;
    ++task_index;
  }

  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
      std::cout << "Process " << rank << ": Thread joined" << std::endl;
    }
  }

  // Собираем результаты от всех процессов
  std::vector<std::vector<double>> all_p(7, std::vector<double>(half_size_squared));
  if (rank == 0) {
    all_p[0] = p1;
    all_p[1] = p2;
    all_p[2] = p3;
    all_p[3] = p4;
    all_p[4] = p5;
    all_p[5] = p6;
    all_p[6] = p7;
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    comm.barrier();
    std::cout << "Process " << rank << ": Broadcasting p" << i + 1 << " from root " << (i % size_comm) << std::endl;
    boost::mpi::broadcast(comm, all_p[i], i % size_comm);
    std::cout << "Process " << rank << ": Broadcasted/received p" << i + 1 << std::endl;
  }

  p1 = all_p[0];
  p2 = all_p[1];
  p3 = all_p[2];
  p4 = all_p[3];
  p5 = all_p[4];
  p6 = all_p[5];
  p7 = all_p[6];

  std::cout << "Process " << rank << ": Computing result matrices" << std::endl;
  std::vector<double> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  std::vector<double> c12 = AddMatrices(p3, p5, half_size);
  std::vector<double> c21 = AddMatrices(p2, p4, half_size);
  std::vector<double> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

  std::vector<double> result(size * size);
  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);

  std::cout << "Process " << rank << ": StrassenMultiply completed" << std::endl;
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
  std::cout << "Process " << boost::mpi::communicator().rank()
            << ": SplitMatrix completed for child size = " << child_size << std::endl;
}

void StrassenAll::MergeMatrix(std::vector<double>& parent, const std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(child.begin() + i * child_size, child.begin() + (i + 1) * child_size,
                      parent.begin() + (row_start + i) * parent_size + col_start);
  }
  std::cout << "Process " << boost::mpi::communicator().rank()
            << ": MergeMatrix completed for child size = " << child_size << std::endl;
}

}  // namespace nasedkin_e_strassen_algorithm_all