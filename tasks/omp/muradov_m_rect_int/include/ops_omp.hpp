#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace muradov_m_rect_int_omp {

using Matfun = std::function<double(const std::vector<double>&)>;
using FunArgs = std::vector<double>;

class RectIntTaskOmp : public ppc::core::Task {
 public:
  explicit RectIntTaskOmp(ppc::core::TaskDataPtr task_data, Matfun fun)
      : Task(std::move(task_data)), fun_(std::move(fun)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  Matfun fun_;
  int grains_;
  std::vector<std::pair<double, double>> bounds_;
  double res_;
};

}  // namespace muradov_m_rect_int_omp