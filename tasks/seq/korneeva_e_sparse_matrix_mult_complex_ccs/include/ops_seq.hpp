#pragma once

#include <complex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace korneeva_e_sparse_matrix_mult_complex_ccs_seq {

using Complex = std::complex<double>;

struct SparseMatrixCCS {
  std::vector<Complex> values;
  std::vector<int> row_indices;
  std::vector<int> col_offsets;
  int rows;
  int cols;
  int nnz;

  SparseMatrixCCS(int r = 0, int c = 0, int n = 0) : rows(r), cols(c), nnz(n) {
    values.resize(nnz);
    row_indices.resize(nnz);
    col_offsets.resize(cols + 1, 0);
  }
};

class SparseMatrixMultComplexCCS : public ppc::core::Task {
 public:
  explicit SparseMatrixMultComplexCCS(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  SparseMatrixCCS* matrix1_;
  SparseMatrixCCS* matrix2_;
  SparseMatrixCCS result_;

  void ComputeColumn(int col_idx, std::vector<Complex>& values, std::vector<int>& row_indices,
                     std::vector<int>& col_offsets);
  Complex ComputeElement(int row_idx, int col_start2, int col_end2);
  Complex ComputeContribution(int row_idx, int k, int col_start1, int col_end1, int col_start2, int col_end2);
};

}  // namespace korneeva_e_sparse_matrix_mult_complex_ccs_seq