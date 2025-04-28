#include "tbb/alputov_i_graham_scan/include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/parallel_sort.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <tuple>
#include <vector>

namespace alputov_i_graham_scan_tbb {

Point::Point(double x, double y) : x(x), y(y) {}

bool Point::operator<(const Point& other) const { return std::tie(y, x) < std::tie(other.y, other.x); }

bool Point::operator==(const Point& other) const { return std::tie(x, y) == std::tie(other.x, other.y); }

bool TestTaskTBB::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<Point*>(task_data->inputs[0]);
  input_points_ = std::vector<Point>(input_ptr, input_ptr + task_data->inputs_count[0]);
  return true;
}

bool TestTaskTBB::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs_count[0] <= task_data->outputs_count[0]);
}

double TestTaskTBB::Cross(const Point& o, const Point& a, const Point& b) {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}

Point TestTaskTBB::FindPivot() const {
  return tbb::parallel_reduce(
      tbb::blocked_range<decltype(input_points_)::const_iterator>(input_points_.begin(), input_points_.end()),
      Point(std::numeric_limits<double>::max(), std::numeric_limits<double>::max()),
      [](const auto& r, Point curr_min) {
        const auto local_min = std::min_element(r.begin(), r.end());
        return (local_min != r.end() && *local_min < curr_min) ? *local_min : curr_min;
      },
      [](const Point& a, const Point& b) { return a < b ? a : b; });
}

void TestTaskTBB::RemoveDuplicates(std::vector<Point>& points) {
  auto result = std::ranges::unique(points);
  points.erase(result.end(), points.end());
}

bool TestTaskTBB::CompareAngles(const Point& first_point, const Point& second_point, const Point& pivot_point) {
  const auto first_dx = first_point.x - pivot_point.x;
  const auto first_dy = first_point.y - pivot_point.y;
  const auto second_dx = second_point.x - pivot_point.x;
  const auto second_dy = second_point.y - pivot_point.y;

  const double cross_product = (first_dx * second_dy) - (first_dy * second_dx);
  constexpr double kEpsilon = 1e-10;

  if (std::abs(cross_product) < kEpsilon) {
    const auto first_distance_squared = (first_dx * first_dx) + (first_dy * first_dy);
    const auto second_distance_squared = (second_dx * second_dx) + (second_dy * second_dy);
    return first_distance_squared < second_distance_squared;
  }

  return cross_product > 0;
}

std::vector<Point> TestTaskTBB::SortPoints(const Point& pivot) const {
  std::vector<Point> points;
  points.reserve(input_points_.size());
  for (const auto& p : input_points_) {
    if (!(p == pivot)) {
      points.push_back(p);
    }
  }

  tbb::parallel_sort(points.begin(), points.end(),
                     [&](const Point& a, const Point& b) { return TestTaskTBB::CompareAngles(a, b, pivot); });
  TestTaskTBB::RemoveDuplicates(points);
  return points;
}

std::vector<Point> TestTaskTBB::BuildHull(const std::vector<Point>& sorted_points, const Point& pivot) {
  std::vector<Point> hull;
  hull.reserve(sorted_points.size() + 1);
  hull.push_back(pivot);

  for (const auto& p : sorted_points) {
    while (hull.size() >= 2 && TestTaskTBB::Cross(hull[hull.size() - 2], hull.back(), p) < 1e-10) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  while (hull.size() >= 3 && TestTaskTBB::Cross(hull[hull.size() - 2], hull.back(), hull[0]) < 1e-10) {
    hull.pop_back();
  }

  return hull;
}

bool TestTaskTBB::RunImpl() {
  const Point pivot = FindPivot();
  const auto sorted_points = SortPoints(pivot);

  if (sorted_points.empty()) {
    convex_hull_ = {pivot};
    return true;
  }

  convex_hull_ = TestTaskTBB::BuildHull(sorted_points, pivot);
  return true;
}

bool TestTaskTBB::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(convex_hull_, output_ptr);
  return true;
}

const std::vector<Point>& TestTaskTBB::GetConvexHull() const { return convex_hull_; }

}  // namespace alputov_i_graham_scan_tbb