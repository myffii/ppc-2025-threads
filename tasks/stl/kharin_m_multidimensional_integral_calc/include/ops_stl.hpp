#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kharin_m_multidimensional_integral_calc_stl {

class TaskSTL : public ppc::core::Task {
 public:
  explicit TaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_;       // Значения функции на сетке
  std::vector<size_t> grid_sizes_;  // Размеры сетки в каждом измерении
  std::vector<double> step_sizes_;  // Шаги интегрирования в каждом измерении
  double output_result_{0.0};       // Результат вычисления интеграла
  size_t num_threads_{1};
};

}  // namespace kharin_m_multidimensional_integral_calc_stl