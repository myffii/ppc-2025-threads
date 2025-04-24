#include "../include/ops_tbb.hpp"

#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

namespace {
double AbsBound(const polikanov_v_rectangles::IntegrationBound& bound) { return bound.second - bound.first; }

std::size_t SizePow(std::size_t a, std::size_t n) {
  std::size_t r = 1;
  for (std::size_t i = 0; i < n; i++) {
    r *= a;
  }
  return r;
}
}  // namespace

class ParallelPointsIterator {
 public:
  ParallelPointsIterator(std::size_t discretization, std::vector<polikanov_v_rectangles::IntegrationBound> bounds)
      : params_({
            .discretization = discretization,
            .bounds = std::move(bounds),
        }),
        points_(SizePow(discretization, params_.bounds.size())) {}

  template <typename AccT, typename ArgsReducer>
  AccT Iterate(tbb::task_arena& arena, ArgsReducer reduce) {
    return arena.execute([&] {
      return oneapi::tbb::parallel_reduce(
          oneapi::tbb::blocked_range<std::size_t>(0, points_, points_ / arena.max_concurrency()), AccT{},
          [&](const auto& r, AccT acc) {
            std::vector<double> point(params_.bounds.size());
            for (std::size_t pt = r.begin(); pt < r.end(); ++pt) {
              std::size_t c{pt};

              auto it = point.begin();
              for (const auto& bound : params_.bounds) {
                *it = bound.first + (static_cast<double>(c % params_.discretization) * AbsBound(bound) /
                                     static_cast<double>(params_.discretization));
                ++it;
                c /= params_.discretization;
              }

              acc += reduce(point);
            }
            return acc;
          },
          std::plus<>());
    });
  }

 private:
  struct Params {
    std::size_t discretization;
    std::vector<std::pair<double, double>> bounds;
  } params_;
  std::size_t points_;
};

namespace polikanov_v_rectangles {

bool polikanov_v_rectangles::TaskTBB::ValidationImpl() {
  return task_data->inputs.size() == 2 && task_data->inputs_count[0] > 0 && task_data->outputs.size() == 1;
}

bool polikanov_v_rectangles::TaskTBB::PreProcessingImpl() {
  const auto* bounds_ptr = reinterpret_cast<polikanov_v_rectangles::IntegrationBound*>(task_data->inputs[0]);
  const auto bounds_size = task_data->inputs_count[0];
  bounds_.assign(bounds_ptr, bounds_ptr + bounds_size);

  discretization_ = *reinterpret_cast<std::size_t*>(task_data->inputs[1]);

  return true;
}

bool polikanov_v_rectangles::TaskTBB::RunImpl() {
  ParallelPointsIterator iter(discretization_, bounds_);

  tbb::task_arena arena{ppc::util::GetPPCNumThreads()};
  result_ = iter.Iterate<double>(arena, [&](std::vector<double>& point) { return function_(point); });
  for (const auto& bound : bounds_) {
    result_ *= AbsBound(bound) / static_cast<double>(discretization_);
  }

  return true;
}

bool polikanov_v_rectangles::TaskTBB::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
  return true;
}
}  // namespace polikanov_v_rectangles
