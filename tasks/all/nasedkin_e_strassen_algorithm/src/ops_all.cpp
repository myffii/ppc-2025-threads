#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <functional>
#include <thread>

#include "core/util/include/util.hpp"

namespace nasedkin_e_strassen_algorithm_all {

boost::mpi::communicator StrassenAll::comm;

bool StrassenAll::PreProcessingImpl() {
  auto input_size = task_data->inputs_count[0];
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
  auto input_size_a = task_data->inputs_count[0];
  auto input_size_b = task_data->inputs_count[1];
  auto output_size = task_data->outputs_count[0];

  if (input_size_a == 0 || input_size_b == 0 || output_size == 0) {
    return false;
  }

  int size_a = static_cast<int>(std::sqrt(input_size_a));
  int size_b = static_cast<int>(std::sqrt(input_size_b));
  int size_output = static_cast<int>(std::sqrt(output_size));

  return (size_a == size_b) && (size_a == size_output);
}

bool StrassenAll::RunImpl() {
  size_t num_threads = std::min<size_t>(16, ppc::util::GetPPCNumThreads());
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_, num_threads, true);
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
        result[i * size + j] += a[i * size + k] * b[k * size + j];
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
                                                  size_t num_threads, bool top_level) {
  if (size <= 32) {
    return StandardMultiply(a, b, size);
  }

  int half_size = size / 2;
  int half_size_squared = half_size * half_size;

  std::vector<double> a11(half_size_squared), a12(half_size_squared), a21(half_size_squared), a22(half_size_squared);
  std::vector<double> b11(half_size_squared), b12(half_size_squared), b21(half_size_squared), b22(half_size_squared;
  SplitMatrix(a, a11, 0, 0, size);
  SplitMatrix(a, a12, 0, half_size, size);
  SplitMatrix(a, a21, half_size, 0, size);
  SplitMatrix(a, a22, half_size, half_size, size);
  SplitMatrix(b, b11, 0, 0, size);
  SplitMatrix(b, b12, 0, half_size, size);
  SplitMatrix(b, b21, half_size, 0, size);
  SplitMatrix(b, b22, half_size, half_size, size);

  if (top_level && comm.size() > 1) {
    int rank = comm.rank();
    int num_procs = comm.size();
    int num_tasks = 7;
    int tasks_per_process = num_tasks / num_procs;
    int extra_tasks = num_tasks % num_procs;
    int start_task = rank * tasks_per_process + std::min(rank, extra_tasks);
    int num_tasks_for_rank = tasks_per_process + (rank < extra_tasks ? 1 : 0);
    int end_task = start_task + num_tasks_for_rank;
    if (end_task > num_tasks) end_task = num_tasks;

    std::vector<std::vector<double>> my_p;
    for (int t = start_task; t < end_task; ++t) {
      std::vector<double> p_i(half_size_squared);
      switch (t) {
        case 0:
          p_i = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size,
                                 num_threads, false);
          break;
        case 1:
          p_i = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads, false);
          break;
        case 2:
          p_i = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads, false);
          break;
        case 3:
          p_i = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads, false);
          break;
        case 4:
          p_i = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads, false);
          break;
        case 5:
          p_i = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                                 num_threads, false);
          break;
        case 6:
          p_i = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                                 num_threads, false);
          break;
      }
      my_p.push_back(p_i);
    }

    if (rank != 0) {
      for (size_t t = 0; t < my_p.size(); ++t) {  // Исправлено: int -> size_t
        comm.send(0, start_task + t, my_p[t]);
      }
      return std::vector<double>();
    } else {
      std::vector<std::vector<double>> p(num_tasks, std::vector<double>(half_size_squared));
      for (size_t t = 0; t < my_p.size(); ++t) {  // Исправлено: int -> size_t
        p[start_task + t] = my_p[t];
      }
      for (int r = 1; r < num_procs; ++r) {
        int r_start_task = r * tasks_per_process + std::min(r, extra_tasks);
        int r_num_tasks = tasks_per_process + (r < extra_tasks ? 1 : 0);
        for (int t = 0; t < r_num_tasks; ++t) {
          int task_index = r_start_task + t;
          if (task_index < num_tasks) {
            comm.recv(r, task_index, p[task_index]);
          }
        }
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
  } else {
    std::vector<std::vector<double>> p(num_tasks, std::vector<double>(half_size_squared));
    std::vector<std::function<void()>> tasks;
    tasks.emplace_back([&]() {
      p[0] = StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size,
                              num_threads, false);
    });
    tasks.emplace_back(
        [&]() { p[1] = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size, num_threads, false); });
    tasks.emplace_back(
        [&]() { p[2] = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size, num_threads, false); });
    tasks.emplace_back(
        [&]() { p[3] = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size, num_threads, false); });
    tasks.emplace_back(
        [&]() { p[4] = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size, num_threads, false); });
    tasks.emplace_back([&]() {
      p[5] = StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size,
                              num_threads, false);
    });
    tasks.emplace_back([&]() {
      p[6] = StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size,
                              num_threads, false);
    });

    std::vector<std::thread> threads;
    threads.reserve(std::min(num_threads, tasks.size()));
    size_t task_index = 0;  // Исправлено: int -> size_t
    for (size_t i = 0; i < std::min(num_threads, tasks.size()); ++i) {
      if (task_index < tasks.size()) {  // Исправлено: int -> size_t
        threads.emplace_back(tasks[task_index]);
        ++task_index;
      }
    }
    while (task_index < tasks.size()) {  // Исправлено: int -> size_t
      tasks[task_index]();
      ++task_index;
    }
    for (auto& thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
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