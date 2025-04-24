#ifndef TASKS_OMP_DURYNICHEV_D_INTEGRALS_SIMPSON_METHOD_INCLUDE_OPS_OMP_HPP_
#define TASKS_OMP_DURYNICHEV_D_INTEGRALS_SIMPSON_METHOD_INCLUDE_OPS_OMP_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace durynichev_d_integrals_simpson_method_omp {

class SimpsonIntegralOpenMP : public ppc::core::Task {
 public:
  enum class FunctionType : std::uint8_t { kSquare, kSin, kCos, kExp, kLog, kCombined };

  explicit SimpsonIntegralOpenMP(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] double GetResult() const { return result_; }
  [[nodiscard]] size_t GetDimension() const { return dim_; }
  [[nodiscard]] int GetNumIntervals() const { return n_; }
  [[nodiscard]] FunctionType GetFunctionType() const { return func_type_; }

 private:
  std::vector<double> boundaries_;
  int n_{};
  size_t dim_{};
  double result_{};
  FunctionType func_type_{FunctionType::kSquare};  // Default function type

  static double Func1DSquare(double x);
  static double Func1DSin(double x);
  static double Func1DCos(double x);
  static double Func1DExp(double x);
  static double Func1DLog(double x);
  static double Func1DCombined(double x);

  static double Func2DSquare(double x, double y);
  static double Func2DSin(double x, double y);
  static double Func2DCos(double x, double y);
  static double Func2DExp(double x, double y);
  static double Func2DLog(double x, double y);
  static double Func2DCombined(double x, double y);

  static double Func3DSquare(double x, double y, double z);
  static double Func3DSin(double x, double y, double z);
  static double Func3DCos(double x, double y, double z);
  static double Func3DExp(double x, double y, double z);
  static double Func3DLog(double x, double y, double z);
  static double Func3DCombined(double x, double y, double z);

  static double GetSimpsonCoefficient(int index, int n);
  double ComputeZIntegral(double x, double y, double z0, double z1, int n, double hz);

  [[nodiscard]] double Simpson1D(double a, double b) const;
  double Simpson2D(double x0, double x1, double y0, double y1);
  double Simpson3D(double x0, double x1, double y0, double y1, double z0, double z1);

  // Evaluate functions based on the function type
  [[nodiscard]] double Evaluate1D(double x) const;
  [[nodiscard]] double Evaluate2D(double x, double y) const;
  [[nodiscard]] double Evaluate3D(double x, double y, double z) const;
};

}  // namespace durynichev_d_integrals_simpson_method_omp

#endif  // TASKS_OMP_DURYNICHEV_D_INTEGRALS_SIMPSON_METHOD_INCLUDE_OPS_OMP_HPP_