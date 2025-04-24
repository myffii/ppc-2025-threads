#include "../include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace {
bool CheckCollinearity(std::span<double> raw_points) {
  const auto points_count = raw_points.size() / 2;
  if (points_count < 3) {
    return true;
  }
  const auto dx = raw_points[2] - raw_points[0];
  const auto dy = raw_points[3] - raw_points[1];
  for (size_t i = 2; i < points_count; i++) {
    const auto dx_i = raw_points[(i * 2)] - raw_points[0];
    const auto dy_i = raw_points[(i * 2) + 1] - raw_points[1];
    if (std::fabs((dx * dy_i) - (dy * dx_i)) > 1e-9) {
      return false;
    }
  }
  return true;
}
bool ComparePoints(const Point &p0, const Point &p1, const Point &p2) {
  const auto dx1 = p1[0] - p0[0];
  const auto dy1 = p1[1] - p0[1];
  const auto dx2 = p2[0] - p0[0];
  const auto dy2 = p2[1] - p0[1];
  const auto cross = (dx1 * dy2) - (dy1 * dx2);
  if (std::abs(cross) < 1e-9) {
    return (dx1 * dx1 + dy1 * dy1) < (dx2 * dx2 + dy2 * dy2);
  }
  return cross > 0;
}
double CrossProduct(const Point &p0, const Point &p1, const Point &p2) {
  return ((p1[0] - p0[0]) * (p2[1] - p0[1])) - ((p1[1] - p0[1]) * (p2[0] - p0[0]));
}
}  // namespace

namespace shvedova_v_graham_convex_hull_seq {

bool GrahamConvexHullSequential::ValidationImpl() {
  return (task_data->inputs.size() == 1 && task_data->inputs_count.size() == 1 && task_data->outputs.size() == 2 &&
          task_data->outputs_count.size() == 2 && (task_data->inputs_count[0] % 2 == 0) &&
          (task_data->inputs_count[0] / 2 > 2) && (task_data->outputs_count[0] == 1) &&
          (task_data->outputs_count[1] >= task_data->inputs_count[0])) &&
         !CheckCollinearity({reinterpret_cast<double *>(task_data->inputs[0]), task_data->inputs_count[0]});
}

bool GrahamConvexHullSequential::PreProcessingImpl() {
  points_count_ = static_cast<int>(task_data->inputs_count[0] / 2);
  input_.resize(points_count_, Point{});

  auto *p_src = reinterpret_cast<double *>(task_data->inputs[0]);
  for (int i = 0; i < points_count_ * 2; i += 2) {
    input_[i / 2][0] = p_src[i];
    input_[i / 2][1] = p_src[i + 1];
  }

  res_.clear();
  res_.reserve(points_count_);

  return true;
}

void GrahamConvexHullSequential::PerformSort() {
  const auto pivot = *std::ranges::min_element(input_, [](auto &a, auto &b) { return a[1] < b[1]; });
  std::ranges::sort(input_.begin(), input_.end(),
                    [&](const Point &p1, const Point &p2) { return ComparePoints(pivot, p1, p2); });
}

bool GrahamConvexHullSequential::RunImpl() {
  PerformSort();
  res_.push_back(input_[0]);
  res_.push_back(input_[1]);
  for (int i = 2; i < points_count_; ++i) {
    while (res_.size() > 1 && CrossProduct(res_[res_.size() - 2], res_.back(), input_[i]) <= 0) {
      res_.pop_back();
    }
    res_.push_back(input_[i]);
  }
  return true;
}

bool GrahamConvexHullSequential::PostProcessingImpl() {
  int res_points_count = static_cast<int>(res_.size());
  *reinterpret_cast<int *>(task_data->outputs[0]) = res_points_count;
  auto *p_out = reinterpret_cast<double *>(task_data->outputs[1]);
  for (int i = 0; i < res_points_count; i++) {
    p_out[2 * i] = res_[i][0];
    p_out[(2 * i) + 1] = res_[i][1];
  }
  return true;
}

}  // namespace shvedova_v_graham_convex_hull_seq
