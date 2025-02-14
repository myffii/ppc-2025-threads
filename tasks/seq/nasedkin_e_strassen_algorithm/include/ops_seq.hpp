#pragma once

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
  std::vector<std::vector<int>> addMatrices(const std::vector<std::vector<int>>& A,
                                            const std::vector<std::vector<int>>& B);
  std::vector<std::vector<int>> subtractMatrices(const std::vector<std::vector<int>>& A,
                                                 const std::vector<std::vector<int>>& B);
  void splitMatrix(const std::vector<std::vector<int>>& parent, std::vector<std::vector<int>>& child, int rowStart,
                   int colStart);
  void mergeMatrix(std::vector<std::vector<int>>& parent, const std::vector<std::vector<int>>& child, int rowStart,
                   int colStart);
  std::vector<std::vector<int>> standardMultiply(const std::vector<std::vector<int>>& A,
                                                 const std::vector<std::vector<int>>& B);
  std::vector<std::vector<int>> strassenMultiply(const std::vector<std::vector<int>>& A,
                                                 const std::vector<std::vector<int>>& B);

  // Данные задачи
  std::vector<std::vector<int>> input_matrix_A_, input_matrix_B_;
  std::vector<std::vector<int>> output_matrix_;
  int matrix_size_{};
};

}  // namespace nasedkin_e_strassen_algorithm_seq