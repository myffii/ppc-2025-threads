#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <cmath>
#include <functional>
#include <future>
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
  boost::mpi::communicator comm;
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_, comm);
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
                                                  boost::mpi::communicator comm) {
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

  int rank = comm.rank();
  int num_procs = comm.size();
  int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::future<void>> futures;

  auto run_task = [&](std::vector<double>& result, const std::vector<double>& ma, const std::vector<double>& mb) {
    result = StrassenMultiply(ma, mb, half_size, comm);
  };

  if (num_procs > 1 && rank < 7) {
    std::vector<std::vector<double>> tasks_a(7), tasks_b(7);
    if (rank == 0) {
      tasks_a[0] = AddMatrices(a11, a22, half_size);
      tasks_b[0] = AddMatrices(b11, b22, half_size);
      tasks_a[1] = AddMatrices(a21, a22, half_size);
      tasks_b[1] = b11;
      tasks_a[2] = a11;
      tasks_b[2] = SubtractMatrices(b12, b22, half_size);
      tasks_a[3] = a22;
      tasks_b[3] = SubtractMatrices(b21, b11, half_size);
      tasks_a[4] = AddMatrices(a11, a12, half_size);
      tasks_b[4] = b22;
      tasks_a[5] = SubtractMatrices(a21, a11, half_size);
      tasks_b[5] = AddMatrices(b11, b12, half_size);
      tasks_a[6] = SubtractMatrices(a12, a22, half_size);
      tasks_b[6] = AddMatrices(b21, b22, half_size);

      for (int i = 1; i < 7 && i < num_procs; ++i) {
        boost::mpi::request req_a = comm.isend(i, 0, tasks_a[i]);
        boost::mpi::request req_b = comm.isend(i, 1, tasks_b[i]);
        req_a.wait();
        req_b.wait();
      }
      p1 = StrassenMultiply(tasks_a[0], tasks_b[0], half_size, comm);
    } else if (rank < 7) {
      comm.recv(0, 0, tasks_a[rank]);
      comm.recv(0, 1, tasks_b[rank]);
      std::vector<double> result = StrassenMultiply(tasks_a[rank], tasks_b[rank], half_size, comm);
      comm.send(0, rank, result);
    }

    if (rank == 0) {
      for (int i = 1; i < 7 && i < num_procs; ++i) {
        std::vector<double> result;
        comm.recv(i, i, result);
        switch (i) {
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
      if (num_procs < 7) {
        for (int i = num_procs; i < 7; ++i) {
          futures.push_back(std::async(std::launch::async, run_task, std::ref(*(&p1 + i)), tasks_a[i], tasks_b[i]));
        }
      }
    }
  } else if (num_procs == 1) {
    std::vector<std::vector<double>> tasks_a = {AddMatrices(a11, a22, half_size),
                                                AddMatrices(a21, a22, half_size),
                                                a11,
                                                a22,
                                                AddMatrices(a11, a12, half_size),
                                                SubtractMatrices(a21, a11, half_size),
                                                SubtractMatrices(a12, a22, half_size)};
    std::vector<std::vector<double>> tasks_b = {
        AddMatrices(b11, b22, half_size),      b11, SubtractMatrices(b12, b22, half_size),
        SubtractMatrices(b21, b11, half_size), b22, AddMatrices(b11, b12, half_size),
        AddMatrices(b21, b22, half_size)};
    std::vector<std::vector<double>*> results = {&p1, &p2, &p3, &p4, &p5, &p6, &p7};

    for (int i = 0; i < 7; ++i) {
      if (i < num_threads) {
        futures.push_back(std::async(std::launch::async, run_task, std::ref(*results[i]), tasks_a[i], tasks_b[i]));
      } else {
        run_task(*results[i], tasks_a[i], tasks_b[i]);
      }
    }
  }

  for (auto& f : futures) {
    f.wait();
  }

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