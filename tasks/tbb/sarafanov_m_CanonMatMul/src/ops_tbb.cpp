#include "tbb/sarafanov_m_CanonMatMul/include/ops_tbb.hpp"

#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/parallel_for.h"
#include "tbb/sarafanov_m_CanonMatMul/include/CanonMatrix.hpp"

// comment to start tests

bool sarafanov_m_canon_mat_mul_tbb::CanonMatMulTBB::PreProcessingImpl() {
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

std::vector<double> sarafanov_m_canon_mat_mul_tbb::CanonMatMulTBB::ConvertToSquareMatrix(
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
      matrix.resize(matrix.size() + (zero_rows * need_size));
      break;
  }
  return matrix;
}

bool sarafanov_m_canon_mat_mul_tbb::CanonMatMulTBB::CheckSquareSize(int number) {
  return task_data->inputs_count[number] == task_data->inputs_count[number + 1];
}

bool sarafanov_m_canon_mat_mul_tbb::CanonMatMulTBB::ValidationImpl() {
  return std::max(task_data->inputs_count[0], task_data->inputs_count[1]) *
             std::max(task_data->inputs_count[2], task_data->inputs_count[3]) ==
         task_data->outputs_count[0];
}

bool sarafanov_m_canon_mat_mul_tbb::CanonMatMulTBB::RunImpl() {
  c_matrix_.ClearMatrix();
  std::vector<CanonMatrix> mul_results(ppc::util::GetPPCNumThreads());
  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  arena.execute([&] {
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<size_t>(0, a_matrix_.GetSize(), a_matrix_.GetSize() / ppc::util::GetPPCNumThreads()),
        [&](const oneapi::tbb::blocked_range<size_t> &distance) {
          for (auto i = distance.begin(); i != distance.end(); ++i) {
            mul_results[tbb::this_task_arena::current_thread_index()] += a_matrix_.MultiplicateMatrix(b_matrix_, i);
          }
        });
  });
  std::ranges::for_each(mul_results, [&](const auto &vector) {
    if (!vector.IsEmpty()) {
      c_matrix_ += vector;
    }
  });
  return true;
}

bool sarafanov_m_canon_mat_mul_tbb::CanonMatMulTBB::PostProcessingImpl() {
  std::ranges::copy(c_matrix_.GetMatrix(), reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
