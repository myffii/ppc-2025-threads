#include "omp/korneeva_e_sparse_matrix_mult_complex_ccs/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <utility>
#include <vector>

namespace korneeva_e_sparse_matrix_mult_complex_ccs_omp {

bool SparseMatrixMultComplexCCS::PreProcessingImpl() {
  matrix1_ = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[0]);
  matrix2_ = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[1]);
  result_ = SparseMatrixCCS(matrix1_->rows, matrix2_->cols, 0);
  return true;
}

bool SparseMatrixMultComplexCCS::ValidationImpl() {
  return task_data->inputs.size() == 2 && task_data->outputs.size() == 1 && matrix1_ != nullptr &&
         matrix2_ != nullptr && matrix1_->cols == matrix2_->rows && matrix1_->rows > 0 && matrix1_->cols > 0 &&
         matrix2_->rows > 0 && matrix2_->cols > 0;
}

bool SparseMatrixMultComplexCCS::RunImpl() {
  std::vector<std::vector<Complex>> local_values(matrix2_->cols);
  std::vector<std::vector<int>> local_row_indices(matrix2_->cols);
  std::vector<int> temp_col_offsets(matrix2_->cols + 1, 0);

#pragma omp parallel for
  for (int j = 0; j < matrix2_->cols; j++) {
    ComputeColumn(j, local_values[j], local_row_indices[j], temp_col_offsets);
  }

  std::vector<Complex> final_values;
  std::vector<int> final_row_indices;
  temp_col_offsets[0] = 0;

  for (int j = 0; j < matrix2_->cols; j++) {
    final_values.insert(final_values.end(), local_values[j].begin(), local_values[j].end());
    final_row_indices.insert(final_row_indices.end(), local_row_indices[j].begin(), local_row_indices[j].end());
    temp_col_offsets[j + 1] = static_cast<int>(final_values.size());
  }

  result_.values = std::move(final_values);
  result_.row_indices = std::move(final_row_indices);
  result_.col_offsets = std::move(temp_col_offsets);
  result_.nnz = static_cast<int>(result_.values.size());
  return true;
}

void SparseMatrixMultComplexCCS::ComputeColumn(int col_idx, std::vector<Complex>& values, std::vector<int>& row_indices,
                                               std::vector<int>& col_offsets) {
  int col_start2 = matrix2_->col_offsets[col_idx];
  int col_end2 = matrix2_->col_offsets[col_idx + 1];

  for (int i = 0; i < matrix1_->rows; i++) {
    Complex sum = ComputeElement(i, col_start2, col_end2);
    if (sum != Complex(0.0, 0.0)) {
      values.push_back(sum);
      row_indices.push_back(i);
    }
  }
}

Complex SparseMatrixMultComplexCCS::ComputeElement(int row_idx, int col_start2, int col_end2) {
  Complex sum(0.0, 0.0);
  for (int k = 0; k < matrix1_->cols; k++) {
    int col_start1 = matrix1_->col_offsets[k];
    int col_end1 = matrix1_->col_offsets[k + 1];
    sum += ComputeContribution(row_idx, k, col_start1, col_end1, col_start2, col_end2);
  }
  return sum;
}

Complex SparseMatrixMultComplexCCS::ComputeContribution(int row_idx, int k, int col_start1, int col_end1,
                                                        int col_start2, int col_end2) {
  Complex contribution(0.0, 0.0);
  for (int p = col_start1; p < col_end1; p++) {
    if (matrix1_->row_indices[p] == row_idx) {
      for (int q = col_start2; q < col_end2; q++) {
        if (matrix2_->row_indices[q] == k) {
          contribution += matrix1_->values[p] * matrix2_->values[q];
        }
      }
    }
  }
  return contribution;
}

bool SparseMatrixMultComplexCCS::PostProcessingImpl() {
  *reinterpret_cast<SparseMatrixCCS*>(task_data->outputs[0]) = result_;
  return true;
}
}  // namespace korneeva_e_sparse_matrix_mult_complex_ccs_omp