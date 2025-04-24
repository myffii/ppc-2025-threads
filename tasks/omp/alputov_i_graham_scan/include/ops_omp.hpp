#pragma once

#include <omp.h>

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace alputov_i_graham_scan_omp {

struct Point {
  double x, y;
  Point(double x = 0, double y = 0);
  bool operator<(const Point& other) const;
  bool operator==(const Point& other) const;
};

class TestTaskOMP : public ppc::core::Task {
 public:
  explicit TestTaskOMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] const std::vector<Point>& GetConvexHull() const;

 private:
  std::vector<Point> input_points_;
  std::vector<Point> convex_hull_;

  [[nodiscard]] Point FindPivot() const;
  [[nodiscard]] std::vector<Point> SortPoints(const Point& pivot) const;
  [[nodiscard]] std::vector<Point> BuildHull(const std::vector<Point>& sorted_points) const;

  static double Cross(const Point& o, const Point& a, const Point& b);
};

}  // namespace alputov_i_graham_scan_omp