#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace petrov_o_vertical_image_filtration_omp {

class TaskOpenMP : public ppc::core::Task {
 public:
  explicit TaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  int width_{}, height_{};
  std::vector<float> gaussian_kernel_ = {1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F, 2.0F / 16.0F, 4.0F / 16.0F,
                                         2.0F / 16.0F, 1.0F / 16.0F, 2.0F / 16.0F, 1.0F / 16.0F};
};

}  // namespace petrov_o_vertical_image_filtration_omp
