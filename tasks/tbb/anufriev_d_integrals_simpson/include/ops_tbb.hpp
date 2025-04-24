#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace anufriev_d_integrals_simpson_tbb {

class IntegralsSimpsonTBB : public ppc::core::Task {
 public:
  explicit IntegralsSimpsonTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int dimension_{};

  std::vector<double> a_, b_;
  std::vector<int> n_;
  int func_code_{};
  double result_{};

  [[nodiscard]] double FunctionN(const std::vector<double>& coords) const;
};

}  // namespace anufriev_d_integrals_simpson_tbb