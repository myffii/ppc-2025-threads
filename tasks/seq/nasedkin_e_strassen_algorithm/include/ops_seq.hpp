#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm_seq {

class StrassenSequential : public ppc::core::Task {
 public:
  explicit StrassenSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  // Вспомогательные методы для алгоритма Штрассена
  static std::vector<int> AddMatrices(const std::vector<int>& a, const std::vector<int>& b);
  static std::vector<int> SubtractMatrices(const std::vector<int>& a, const std::vector<int>& b);
  static void SplitMatrix(const std::vector<int>& parent, std::vector<int>& child, int row_start, int col_start);
  static void MergeMatrix(std::vector<int>& parent, const std::vector<int>& child, int row_start, int col_start);
  static std::vector<int> StandardMultiply(const std::vector<int>& a, const std::vector<int>& b);
  static std::vector<int> PadMatrixToPowerOfTwo(const std::vector<int>& matrix);
  static std::vector<int> TrimMatrixToOriginalSize(const std::vector<int>& matrix, size_t original_size);
  static std::vector<int> StrassenMultiply(const std::vector<int>& a, const std::vector<int>& b);

  // Данные задачи
  std::vector<int> input_matrix_a_, input_matrix_b_;
  std::vector<int> output_matrix_;
  int matrix_size_{};
  int original_size_{};
};

}  // namespace nasedkin_e_strassen_algorithm_seq