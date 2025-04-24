#include "omp/alputov_i_graham_scan/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace alputov_i_graham_scan_omp {

Point::Point(double x, double y) : x(x), y(y) {}

bool Point::operator<(const Point& other) const { return y < other.y || (y == other.y && x < other.x); }

bool Point::operator==(const Point& other) const { return x == other.x && y == other.y; }

bool TestTaskOMP::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<Point*>(task_data->inputs[0]);
  size_t count = task_data->inputs_count[0];
  input_points_.assign(input_ptr, input_ptr + count);
  return true;
}

bool TestTaskOMP::ValidationImpl() {
  return task_data->inputs_count[0] >= 3 && task_data->inputs_count[0] <= task_data->outputs_count[0];
}

double TestTaskOMP::Cross(const Point& o, const Point& a, const Point& b) {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}

Point TestTaskOMP::FindPivot() const {
  Point pivot = input_points_[0];
#pragma omp parallel
  {
    Point local_pivot = pivot;
#pragma omp for nowait
    for (int i = 1; i < static_cast<int>(input_points_.size()); ++i) {
      if (input_points_[i] < local_pivot) {
        local_pivot = input_points_[i];
      }
    }
#pragma omp critical
    {
      if (local_pivot < pivot) {
        pivot = local_pivot;
      }
    }
  }
  return pivot;
}

std::vector<Point> TestTaskOMP::SortPoints(const Point& pivot) const {
  std::vector<Point> points;
  points.reserve(input_points_.size());

  std::vector<std::vector<Point>> local_points(omp_get_max_threads());

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
#pragma omp for nowait
    for (int i = 0; i < static_cast<int>(input_points_.size()); ++i) {
      if (!(input_points_[i] == pivot)) {
        local_points[tid].push_back(input_points_[i]);
      }
    }
  }

  for (const auto& vec : local_points) {
    points.insert(points.end(), vec.begin(), vec.end());
  }

  std::ranges::sort(points, [&pivot](const Point& a, const Point& b) {
    double angle_a = atan2(a.y - pivot.y, a.x - pivot.x);
    double angle_b = atan2(b.y - pivot.y, b.x - pivot.x);
    if (angle_a != angle_b) {
      return angle_a < angle_b;
    }
    return (a.x - pivot.x) * (a.x - pivot.x) + (a.y - pivot.y) * (a.y - pivot.y) <
           (b.x - pivot.x) * (b.x - pivot.x) + (b.y - pivot.y) * (b.y - pivot.y);
  });

  auto unique_result = std::ranges::unique(points);
  points.erase(unique_result.end(), points.end());

  return points;
}

std::vector<Point> TestTaskOMP::BuildHull(const std::vector<Point>& sorted_points) const {
  std::vector<Point> hull;
  if (sorted_points.size() < 2) {
    return sorted_points;
  }

  hull.reserve(sorted_points.size());
  hull.push_back(FindPivot());
  hull.push_back(sorted_points[0]);

  for (size_t i = 1; i < sorted_points.size(); ++i) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), sorted_points[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(sorted_points[i]);
  }
  return hull;
}

bool TestTaskOMP::RunImpl() {
  const Point pivot = FindPivot();
  const auto sorted_points = SortPoints(pivot);
  convex_hull_ = BuildHull(sorted_points);
  return true;
}

bool TestTaskOMP::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(convex_hull_, output_ptr);
  return true;
}

const std::vector<Point>& TestTaskOMP::GetConvexHull() const { return convex_hull_; }

}  // namespace alputov_i_graham_scan_omp