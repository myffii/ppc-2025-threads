#include "omp/durynichev_d_integrals_simpson_method/include/ops_omp.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

bool durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  boundaries_ = std::vector<double>(in_ptr, in_ptr + input_size);

  if (input_size >= 4) {
    // Last element - function type
    func_type_ = static_cast<FunctionType>(static_cast<int>(boundaries_.back()));
    boundaries_.pop_back();
  } else {
    // By default, use square function
    func_type_ = FunctionType::kSquare;
  }

  n_ = static_cast<int>(boundaries_.back());
  boundaries_.pop_back();

  // Determine the number of dimensions based on the number of boundaries
  dim_ = static_cast<size_t>(boundaries_.size() / 2);

  result_ = 0.0;
  return true;
}

bool durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::ValidationImpl() {
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  if (task_data->inputs_count[0] < 3 || task_data->outputs_count[0] != 1) {
    return false;
  }

  int n = static_cast<int>(in_ptr[task_data->inputs_count[0] - 2]);

  return (n % 2 == 0);
}

bool durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::RunImpl() {
  if (dim_ == 1) {
    result_ = Simpson1D(boundaries_[0], boundaries_[1]);
  } else if (dim_ == 2) {
    result_ = Simpson2D(boundaries_[0], boundaries_[1], boundaries_[2], boundaries_[3]);
  } else if (dim_ == 3) {
    result_ = Simpson3D(boundaries_[0], boundaries_[1], boundaries_[2], boundaries_[3], boundaries_[4], boundaries_[5]);
  }
  return true;
}

bool durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}

// 1D
double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func1DSquare(double x) { return x * x; }

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func1DSin(double x) { return std::sin(x); }

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func1DCos(double x) { return std::cos(x); }

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func1DExp(double x) {
  // Restricting the value of x to avoid overflow
  if (x > 700.0) {
    return std::numeric_limits<double>::max() / 2.0;  // Return a large but not infinite value
  }
  return std::exp(x);
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func1DLog(double x) {
  return std::log(std::abs(x) + 1e-10);  // Avoid log(0)
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func1DCombined(double x) {
  return std::sin(x) + std::cos(x) + (x * x);
}

// 2D
double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func2DSquare(double x, double y) {
  return (x * x) + (y * y);
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func2DSin(double x, double y) {
  return std::sin(x) * std::sin(y);
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func2DCos(double x, double y) {
  return std::cos(x) * std::cos(y);
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func2DExp(double x, double y) {
  // Restricting the sum x+y to avoid overflow
  double sum = x + y;
  if (sum > 700.0) {
    return std::numeric_limits<double>::max() / 2.0;  // Return a large but not infinite value
  }
  return std::exp(sum);
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func2DLog(double x, double y) {
  return std::log(std::abs(x * y) + 1e-10);  // Avoid log(0)
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func2DCombined(double x, double y) {
  return std::sin(x) + std::cos(y) + (x * x) + (y * y);
}

// 3D
double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func3DSquare(double x, double y, double z) {
  return (x * x) + (y * y) + (z * z);
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func3DSin(double x, double y, double z) {
  return std::sin(x) * std::sin(y) * std::sin(z);
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func3DCos(double x, double y, double z) {
  return std::cos(x) * std::cos(y) * std::cos(z);
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func3DExp(double x, double y, double z) {
  // Restricting the sum x+y+z to avoid overflow
  double sum = x + y + z;
  if (sum > 700.0) {
    return std::numeric_limits<double>::max() / 2.0;
  }
  return std::exp(sum);
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func3DLog(double x, double y, double z) {
  return std::log(std::abs(x * y * z) + 1e-10);  // Avoid log(0)
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func3DCombined(double x, double y, double z) {
  return std::sin(x) + std::cos(y) + std::sin(z) + (x * x) + (y * y) + (z * z);
}

// Function evaluation methods
double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Evaluate1D(double x) const {
  switch (func_type_) {
    case FunctionType::kSquare:
      return Func1DSquare(x);
    case FunctionType::kSin:
      return Func1DSin(x);
    case FunctionType::kCos:
      return Func1DCos(x);
    case FunctionType::kExp:
      return Func1DExp(x);
    case FunctionType::kLog:
      return Func1DLog(x);
    case FunctionType::kCombined:
      return Func1DCombined(x);
    default:
      return Func1DSquare(x);
  }
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Evaluate2D(double x, double y) const {
  switch (func_type_) {
    case FunctionType::kSquare:
      return Func2DSquare(x, y);
    case FunctionType::kSin:
      return Func2DSin(x, y);
    case FunctionType::kCos:
      return Func2DCos(x, y);
    case FunctionType::kExp:
      return Func2DExp(x, y);
    case FunctionType::kLog:
      return Func2DLog(x, y);
    case FunctionType::kCombined:
      return Func2DCombined(x, y);
    default:
      return Func2DSquare(x, y);
  }
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Evaluate3D(double x, double y,
                                                                                    double z) const {
  switch (func_type_) {
    case FunctionType::kSquare:
      return Func3DSquare(x, y, z);
    case FunctionType::kSin:
      return Func3DSin(x, y, z);
    case FunctionType::kCos:
      return Func3DCos(x, y, z);
    case FunctionType::kExp:
      return Func3DExp(x, y, z);
    case FunctionType::kLog:
      return Func3DLog(x, y, z);
    case FunctionType::kCombined:
      return Func3DCombined(x, y, z);
    default:
      return Func3DSquare(x, y, z);
  }
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::GetSimpsonCoefficient(int index, int n) {
  if (index == 0 || index == n) {
    return 1.0;
  }
  return (index % 2 == 0) ? 2.0 : 4.0;
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::ComputeZIntegral(double x, double y, double z0,
                                                                                          double z1, int n, double hz) {
  double local_sum = 0.0;
  for (int k = 0; k <= n; k++) {
    double z = z0 + (k * hz);
    double coef_z = GetSimpsonCoefficient(k, n);
    local_sum += coef_z * Evaluate3D(x, y, z);
  }
  return local_sum;
}

// Simpson integration methods
double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Simpson1D(double a, double b) const {
  double h = (b - a) / n_;
  double sum = Evaluate1D(a) + Evaluate1D(b);
  double sum_odd = 0.0;
  double sum_even = 0.0;

#pragma omp parallel
  {
#pragma omp for reduction(+ : sum_odd)
    for (int i = 1; i < n_; i += 2) {
      sum_odd += Evaluate1D(a + (i * h));
    }

#pragma omp for reduction(+ : sum_even)
    for (int i = 2; i < n_ - 1; i += 2) {
      sum_even += Evaluate1D(a + (i * h));
    }
  }

  sum += 4 * sum_odd + 2 * sum_even;
  return sum * h / 3.0;
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Simpson2D(double x0, double x1, double y0,
                                                                                   double y1) {
  double hx = (x1 - x0) / n_;
  double hy = (y1 - y0) / n_;
  double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i <= n_; i++) {
    double x = x0 + (i * hx);
    double coef_x = GetSimpsonCoefficient(i, n_);
    double local_sum = 0.0;

    for (int j = 0; j <= n_; j++) {
      double y = y0 + (j * hy);
      double coef_y = GetSimpsonCoefficient(j, n_);
      local_sum += coef_y * Evaluate2D(x, y);
    }
    sum += coef_x * local_sum;
  }

  return sum * hx * hy / 9.0;
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Simpson3D(double x0, double x1, double y0,
                                                                                   double y1, double z0, double z1) {
  double hx = (x1 - x0) / n_;
  double hy = (y1 - y0) / n_;
  double hz = (z1 - z0) / n_;
  double sum = 0.0;
  int total_iterations = (n_ + 1) * (n_ + 1);

#pragma omp parallel for reduction(+ : sum)
  for (int idx = 0; idx < total_iterations; idx++) {
    int i = idx / (n_ + 1);
    int j = idx % (n_ + 1);
    double x = x0 + (i * hx);
    double y = y0 + (j * hy);
    double coef_x = GetSimpsonCoefficient(i, n_);
    double coef_y = GetSimpsonCoefficient(j, n_);
    double local_sum = ComputeZIntegral(x, y, z0, z1, n_, hz);
    sum += coef_x * coef_y * local_sum;
  }

  return sum * hx * hy * hz / 27.0;
}
