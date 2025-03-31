#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm_omp {

std::vector<double> StandardMultiply(const std::vector<double>& a, const std::vector<double>& b, int rows_a, int cols_a,
                                     int cols_b);

class StrassenOmp : public ppc::core::Task {
 public:
  explicit StrassenOmp(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static std::vector<double> AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int rows,
                                         int cols);
  static std::vector<double> SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b, int rows,
                                              int cols);
  static void SplitMatrix(const std::vector<double>& parent, std::vector<double>& child, int row_start, int col_start,
                          int parent_rows, int parent_cols);
  static void MergeMatrix(std::vector<double>& parent, const std::vector<double>& child, int row_start, int col_start,
                          int parent_rows, int parent_cols);
  static std::vector<double> PadMatrix(const std::vector<double>& matrix, int orig_rows, int orig_cols, int new_rows,
                                       int new_cols);
  static std::vector<double> TrimMatrix(const std::vector<double>& matrix, int padded_rows, int padded_cols,
                                        int orig_rows, int orig_cols);
  static std::vector<double> StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b, int rows_a,
                                              int cols_a, int cols_b);

  std::vector<double> input_matrix_a_, input_matrix_b_;
  std::vector<double> output_matrix_;
  int rows_a_{}, cols_a_{}, cols_b_{};
  int orig_rows_a_{}, orig_cols_a_{}, orig_cols_b_{};
};
}  // namespace nasedkin_e_strassen_algorithm_omp