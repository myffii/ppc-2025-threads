#pragma once
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace shurigin_s_integrals_square_tbb {
class Integral : public ppc::core::Task {
 public:
  explicit Integral(std::shared_ptr<ppc::core::TaskData> task_data);

  void SetFunction(const std::function<double(double)>& func);
  void SetFunction(const std::function<double(const std::vector<double>&)>& func, int dimensions);

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double ComputeRecursiveTBB(const std::function<double(const std::vector<double>&)>& f, const std::vector<double>& a,
                             const std::vector<double>& b, const std::vector<int>& n, int dim,
                             std::vector<double> point, int current_dim);
  bool ComputeOneDimensional();

  std::vector<double> down_limits_;
  std::vector<double> up_limits_;
  std::vector<int> counts_;
  double result_;
  std::function<double(const std::vector<double>&)> func_;
  int dimensions_;
  std::shared_ptr<ppc::core::TaskData> task_data_;
};

}  // namespace shurigin_s_integrals_square_tbb