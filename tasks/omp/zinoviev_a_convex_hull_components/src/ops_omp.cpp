#include "omp/zinoviev_a_convex_hull_components/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using namespace zinoviev_a_convex_hull_components_omp;

ConvexHullOMP::ConvexHullOMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool ConvexHullOMP::PreProcessingImpl() noexcept {
  if (!task_data || task_data->inputs.empty() || task_data->inputs_count.size() < 2) {
    return false;
  }

  const auto* input_data = reinterpret_cast<int*>(task_data->inputs[0]);
  const int width = static_cast<int>(task_data->inputs_count[0]);
  const int height = static_cast<int>(task_data->inputs_count[1]);
  input_points_.clear();

  std::vector<std::vector<Point>> private_points(omp_get_max_threads());

#pragma omp parallel for
  for (int y = 0; y < height; ++y) {
    const int tid = omp_get_thread_num();
    const auto w = static_cast<size_t>(width);
    for (int x = 0; x < width; ++x) {
      const size_t idx = (static_cast<size_t>(y) * w) + static_cast<size_t>(x);
      if (idx < static_cast<size_t>(width) * static_cast<size_t>(height) && input_data[idx] != 0) {
        private_points[tid].emplace_back(Point{.x = x, .y = y});
      }
    }
  }

  for (auto& vec : private_points) {
    input_points_.insert(input_points_.end(), vec.begin(), vec.end());
  }

  return true;
}

bool ConvexHullOMP::ValidationImpl() noexcept {
  return task_data->inputs_count.size() == 2 && task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0;
}

int ConvexHullOMP::Cross(const Point& o, const Point& a, const Point& b) noexcept {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}

std::vector<Point> ConvexHullOMP::FindConvexHull(const std::vector<Point>& points) noexcept {
  if (points.size() < 3) {
    return points;
  }

  std::vector<Point> sorted(points);
  std::ranges::sort(sorted, std::less{});

  std::vector<Point> hull;
  for (const auto& p : sorted) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  hull.pop_back();
  for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), *it) <= 0) {
      hull.pop_back();
    }
    hull.push_back(*it);
  }

  if (!hull.empty()) {
    hull.pop_back();
  }
  return hull;
}

bool ConvexHullOMP::RunImpl() noexcept {
  output_hull_ = FindConvexHull(input_points_);
  return true;
}

bool ConvexHullOMP::PostProcessingImpl() noexcept {
  if (task_data->outputs.empty() || static_cast<size_t>(task_data->outputs_count[0]) < output_hull_.size()) {
    return false;
  }

  auto* output = reinterpret_cast<Point*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_hull_.size(); ++i) {
    output[i] = output_hull_[i];
  }

  task_data->outputs_count[0] = static_cast<int>(output_hull_.size());
  return true;
}