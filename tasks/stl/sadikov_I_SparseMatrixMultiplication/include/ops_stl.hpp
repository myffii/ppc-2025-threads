#pragma once

#include <cmath>
#include <utility>

#include "core/task/include/task.hpp"
#include "stl/sadikov_I_SparseMatrixMultiplication/include/SparseMatrix.hpp"

namespace sadikov_i_sparse_matrix_multiplication_task_stl {

class CCSMatrixSTL : public ppc::core::Task {
  SparseMatrix m_fMatrix_;
  SparseMatrix m_sMatrix_;
  SparseMatrix m_answerMatrix_;

 public:
  explicit CCSMatrixSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sadikov_i_sparse_matrix_multiplication_task_stl