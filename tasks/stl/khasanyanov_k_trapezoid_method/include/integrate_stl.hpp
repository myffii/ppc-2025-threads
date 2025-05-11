#ifndef INTEGRATE_STL_HPP
#define INTEGRATE_STL_HPP

#include <memory>
#include <utility>

#include "../include/integrator.hpp"
#include "core/task/include/task.hpp"

namespace khasanyanov_k_trapezoid_method_stl {

struct TaskContext {
  IntegrationFunction function;
  IntegrationBounds bounds;
  double precision;
};

class TrapezoidalMethodSTL : public ppc::core::Task {
 public:
  explicit TrapezoidalMethodSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void CreateTaskData(std::shared_ptr<ppc::core::TaskData> &, TaskContext &context, double *);

 private:
  TaskContext data_;
  double res_{};
};

}  // namespace khasanyanov_k_trapezoid_method_stl

#endif