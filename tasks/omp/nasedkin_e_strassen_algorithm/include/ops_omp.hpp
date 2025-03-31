#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm_omp {

std::vector<double> StandardMultiply(const std::vector<double>& a, const std::vector<double>& b, int size_a, int size_b);

class StrassenOmp : public ppc::core::Task {
 public:
  explicit StrassenOmp(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static std::vector<double> AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int size);
  static std::vector<double> SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b, int size);
  static void SplitMatrix(const std::vector<double>& parent, std::vector<double>& child, int row_start, int col_start,
                          int parent_size);
  static void MergeMatrix(std::vector<double>& parent, const std::vector<double>& child, int row_start, int col_start,
                          int parent_size);
  static std::vector<double> PadMatrix(const std::vector<double>& matrix, int original_size, int target_size);
  static std::vector<double> TrimMatrix(const std::vector<double>& matrix, int target_size);
  static std::vector<double> StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b, int size_a, int size_b);

  std::vector<double> input_matrix_a_, input_matrix_b_;
  std::vector<double> output_matrix_;
  int matrix_size_a_{};
  int matrix_size_b_{};
  int max_size_{};
};

}  // namespace nasedkin_e_strassen_algorithm_omp