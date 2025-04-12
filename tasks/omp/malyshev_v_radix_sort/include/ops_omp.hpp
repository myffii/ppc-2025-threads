#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_v_radix_sort_omp {

class RadixSortDoubleOMP : public ppc::core::Task {
 public:
  explicit RadixSortDoubleOMP(ppc::core::TaskDataPtr task_data);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;
  static void ConvertDouble(double& val, bool reverse = false);
};

}  // namespace malyshev_v_radix_sort_omp