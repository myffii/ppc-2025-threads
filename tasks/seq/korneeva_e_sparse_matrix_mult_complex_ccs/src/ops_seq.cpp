#include "seq/korneeva_e_sparse_matrix_mult_complex_ccs/include/ops_seq.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace korneeva_e_sparse_matrix_mult_complex_ccs_seq {

bool SparseMatrixMultComplexCCS::PreProcessingImpl() {
  matrix1_ = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[0]);
  matrix2_ = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[1]);
  result_ = SparseMatrixCCS(matrix1_->rows, matrix2_->cols, 0);
  return true;
}

bool SparseMatrixMultComplexCCS::ValidationImpl() {
  if (task_data->inputs.size() != 2 || task_data->outputs.size() != 1) {
    return false;
  }

  auto* m1 = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[0]);
  auto* m2 = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[1]);

  return m1 != nullptr && m2 != nullptr && m1->cols == m2->rows && m1->rows > 0 && m1->cols > 0 && m2->rows > 0 &&
         m2->cols > 0;
}

bool SparseMatrixMultComplexCCS::RunImpl() {
  std::vector<Complex> temp_values;
  std::vector<int> temp_row_indices;
  std::vector<int> temp_col_offsets(1, 0);

  for (int j = 0; j < matrix2_->cols; j++) {
    ComputeColumn(j, temp_values, temp_row_indices, temp_col_offsets);
  }

  result_.values = std::move(temp_values);
  result_.row_indices = std::move(temp_row_indices);
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
  col_offsets.push_back(static_cast<int>(values.size()));
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
}  // namespace korneeva_e_sparse_matrix_mult_complex_ccs_seq