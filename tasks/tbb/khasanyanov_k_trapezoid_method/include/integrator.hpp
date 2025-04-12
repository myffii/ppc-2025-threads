#ifndef INTEGRATOR_HPP
#define INTEGRATOR_HPP

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/tbb.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>
namespace khasanyanov_k_trapezoid_method_tbb {

enum IntegrationTechnology : std::uint8_t { kSequential, kOpenMP, kTBB, kSTL, kMPI };

using IntegrationFunction = std::function<double(const std::vector<double>&)>;
using Bounds = std::pair<double, double>;
using IntegrationBounds = std::vector<Bounds>;

template <IntegrationTechnology technology>
class Integrator {
  static const int kDefaultSteps, kMaxSteps;

  [[nodiscard]] static double TrapezoidalMethodSequential(const IntegrationFunction& f, const IntegrationBounds& bounds,
                                                          int steps);

  [[nodiscard]] static double TrapezoidalMethodTbb(const IntegrationFunction& f, const IntegrationBounds& bounds,
                                                   int steps);

  [[nodiscard]] static double TrapezoidalMethod(const IntegrationFunction&, const IntegrationBounds&, double, int, int,
                                                auto ref_func);

 public:
  double operator()(const IntegrationFunction&, const IntegrationBounds&, double, int = kDefaultSteps,
                    int = kMaxSteps) const;
};

//----------------------------------------------------------------------------------------------------------

template <IntegrationTechnology technology>
const int Integrator<technology>::kDefaultSteps = 10;

template <IntegrationTechnology technology>
const int Integrator<technology>::kMaxSteps = 250;

template <IntegrationTechnology technology>
double Integrator<technology>::operator()(const IntegrationFunction& f, const IntegrationBounds& bounds,
                                          double precision, int init_steps, int max_steps) const {
  switch (technology) {
    case kSequential:
      return TrapezoidalMethod(f, bounds, precision, init_steps, max_steps, &TrapezoidalMethodSequential);
    case kTBB:
      return TrapezoidalMethod(f, bounds, precision, init_steps, max_steps, &TrapezoidalMethodTbb);
    case kMPI:
    case kOpenMP:
    case kSTL:
    default:
      throw std::runtime_error("Technology not available");
  }
}

template <IntegrationTechnology technology>
double Integrator<technology>::TrapezoidalMethod(const IntegrationFunction& f, const IntegrationBounds& bounds,
                                                 double precision, int init_steps, int max_steps, auto ref_func) {
  int steps = init_steps;
  double prev_result = ref_func(f, bounds, steps);
  while (steps <= max_steps) {
    steps *= 2;
    double current_result = ref_func(f, bounds, steps);
    if (std::abs(current_result - prev_result) < precision) {
      return current_result;
    }
    prev_result = current_result;
  }
  return prev_result;
}

template <IntegrationTechnology technology>
double Integrator<technology>::TrapezoidalMethodSequential(const IntegrationFunction& f,
                                                           const IntegrationBounds& bounds, int steps) {
  const size_t dimension = bounds.size();

  std::vector<double> h(dimension);
  double cell_volume = 1.0;
  for (size_t i = 0; i < dimension; ++i) {
    const auto& [a, b] = bounds[i];
    if (b < a) {
      throw std::runtime_error("Wrong bounds");
    }
    h[i] = (b - a) / steps;
    cell_volume *= h[i];
  }

  int total_points = 1;
  for (size_t i = 0; i < dimension; ++i) {
    total_points *= (steps + 1);
  }

  double total_sum = 0.0;

  for (int idx = 0; idx < total_points; ++idx) {
    std::vector<double> point(dimension);
    int temp = idx;
    int boundary_count = 0;

    for (size_t dim = 0; dim < dimension; ++dim) {
      const int steps_per_dim = steps + 1;
      const int step = temp % steps_per_dim;
      temp /= steps_per_dim;

      const auto& [a, _] = bounds[dim];
      point[dim] = a + step * h[dim];

      if (step == 0 || step == steps) {
        boundary_count++;
      }
    }

    const double weight = std::pow(0.5, boundary_count);
    total_sum += f(point) * weight;
  }

  return total_sum * cell_volume;
}

template <IntegrationTechnology technology>
double Integrator<technology>::TrapezoidalMethodTbb(const IntegrationFunction& f, const IntegrationBounds& bounds,
                                                    int steps) {
  const size_t dimension = bounds.size();

  std::vector<double> h(dimension);
  double cell_volume = 1.0;
  for (size_t i = 0; i < dimension; ++i) {
    const auto& [a, b] = bounds[i];
    if (b < a) {
      throw std::runtime_error("Wrong bounds");
    }
    h[i] = (b - a) / steps;
    cell_volume *= h[i];
  }

  int total_points = 1;
  for (size_t i = 0; i < dimension; ++i) {
    total_points *= (steps + 1);
  }

  struct ReduceData {
    double sum{0};
    std::vector<double> point;
    ReduceData(int dim) : point(dim) {}
  };

  auto result = tbb::parallel_reduce(
      tbb::blocked_range<uint64_t>(0, total_points), ReduceData(dimension),
      [&](const tbb::blocked_range<uint64_t>& r, ReduceData init) {
        init.point.resize(dimension);
        for (uint64_t idx = r.begin(); idx != r.end(); ++idx) {
          uint64_t temp = idx;
          int boundary_count = 0;

          for (size_t dim = 0; dim < dimension; ++dim) {
            const int steps_per_dim = steps + 1;
            const uint64_t step = temp % steps_per_dim;
            temp /= steps_per_dim;

            const auto& [a, _] = bounds[dim];
            init.point[dim] = a + static_cast<int>(step) * h[dim];

            if (step == 0 || static_cast<int>(step) == steps) {
              boundary_count++;
            }
          }

          const double weight = std::pow(0.5, boundary_count);
          init.sum += f(init.point) * weight;
        }
        return init;
      },
      [](ReduceData a, const ReduceData& b) {
        a.sum += b.sum;
        return a;
      },
      tbb::auto_partitioner());

  return result.sum * cell_volume;
}

}  // namespace khasanyanov_k_trapezoid_method_tbb

#endif