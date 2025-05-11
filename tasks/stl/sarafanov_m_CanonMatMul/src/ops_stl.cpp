#include "stl/sarafanov_m_CanonMatMul/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"
#include "stl/sarafanov_m_CanonMatMul/include/CanonMatrix.hpp"

sarafanov_m_canon_mat_mul_stl::ThreadDataAmount::ThreadDataAmount(int threads_count, int matrix_size) {
  step_size = (matrix_size - (matrix_size % threads_count)) / threads_count;
  residual_data_amount = matrix_size % threads_count;
}

bool sarafanov_m_canon_mat_mul_stl::CanonMatMulOMP::PreProcessingImpl() {
  a_matrix_.ClearMatrix();
  b_matrix_.ClearMatrix();
  c_matrix_.ClearMatrix();
  int rows = static_cast<int>(task_data->inputs_count[0]);
  int columns = static_cast<int>(task_data->inputs_count[1]);
  std::vector<double> matrix_a(rows * columns);
  auto *in = reinterpret_cast<double *>(task_data->inputs[0]);
  std::copy(in, in + matrix_a.size(), matrix_a.begin());
  if (!CheckSquareSize(0)) {
    matrix_a = ConvertToSquareMatrix(std::max(rows, columns),
                                     rows > columns ? MatrixType::kRowMatrix : MatrixType::kColumnMatrix, matrix_a);
  }
  a_matrix_.SetBaseMatrix(std::move(matrix_a));
  a_matrix_.PreRoutine(MatrixType::kRowMatrix);
  rows = static_cast<int>(task_data->inputs_count[2]);
  columns = static_cast<int>(task_data->inputs_count[3]);
  std::vector<double> matrix_b(rows * columns);
  in = reinterpret_cast<double *>(task_data->inputs[1]);
  std::copy(in, in + matrix_b.size(), matrix_b.begin());
  if (!CheckSquareSize(2)) {
    matrix_b = ConvertToSquareMatrix(std::max(rows, columns),
                                     rows > columns ? MatrixType::kRowMatrix : MatrixType::kColumnMatrix, matrix_b);
  }
  b_matrix_.SetBaseMatrix(std::move(matrix_b));
  b_matrix_.PreRoutine(MatrixType::kColumnMatrix);
  return true;
}

std::vector<double> sarafanov_m_canon_mat_mul_stl::CanonMatMulOMP::ConvertToSquareMatrix(
    int need_size, MatrixType type, const std::vector<double> &matrx) {
  std::vector<double> matrix;
  int rows_counter = 0;
  int zero_columns = 0;
  switch (type) {
    case MatrixType::kRowMatrix:
      rows_counter = 1;
      zero_columns = need_size - (static_cast<int>(matrx.size()) / need_size);
      for (int i = 0; i < static_cast<int>(matrx.size()); ++i) {
        if ((need_size - zero_columns) * rows_counter - i == 0) {
          rows_counter++;
          for (int j = 0; j < zero_columns; ++j) {
            matrix.emplace_back(0.0);
          }
        }
        matrix.emplace_back(matrx[i]);
      }
      for (int i = 0; i < zero_columns; ++i) {
        matrix.emplace_back(0.0);
      }
      break;
    case MatrixType::kColumnMatrix:
      matrix = matrx;
      int zero_rows = need_size - (static_cast<int>(matrx.size()) / need_size);
      for (int i = 0; i < zero_rows * need_size; ++i) {
        matrix.emplace_back(0.0);
      }
      break;
  }
  return matrix;
}

bool sarafanov_m_canon_mat_mul_stl::CanonMatMulOMP::CheckSquareSize(int number) {
  return task_data->inputs_count[number] == task_data->inputs_count[number + 1];
}

bool sarafanov_m_canon_mat_mul_stl::CanonMatMulOMP::ValidationImpl() {
  return std::max(task_data->inputs_count[0], task_data->inputs_count[1]) *
             std::max(task_data->inputs_count[2], task_data->inputs_count[3]) ==
         task_data->outputs_count[0];
}

bool sarafanov_m_canon_mat_mul_stl::CanonMatMulOMP::RunImpl() {
  c_matrix_.ClearMatrix();
  std::vector<CanonMatrix> mul_results(ppc::util::GetPPCNumThreads());
  std::vector<std::thread> threads(ppc::util::GetPPCNumThreads());
  ThreadDataAmount data_amount(ppc::util::GetPPCNumThreads(), static_cast<int>(a_matrix_.GetSize()));
  auto multiplicator = [&](int start_index, int end_index, size_t thread_index) {
    for (int i = start_index; i < end_index; ++i) {
      mul_results[thread_index] += a_matrix_.MultiplicateMatrix(b_matrix_, i);
    }
  };
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i] =
        std::thread(multiplicator, i * data_amount.step_size,
                    i == threads.size() - 1 ? ((i + 1) * data_amount.step_size) + data_amount.residual_data_amount
                                            : (i + 1) * data_amount.step_size,
                    i);
  }
  for (auto &thread : threads) {
    thread.join();
  }
  std::ranges::for_each(mul_results, [&](const auto &vector) {
    if (!vector.IsEmpty()) {
      c_matrix_ += vector;
    }
  });
  return true;
}

bool sarafanov_m_canon_mat_mul_stl::CanonMatMulOMP::PostProcessingImpl() {
  auto matrix = c_matrix_.GetMatrix();
  std::ranges::copy(matrix, reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
