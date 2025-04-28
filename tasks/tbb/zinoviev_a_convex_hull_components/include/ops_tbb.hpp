#pragma once

#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/concurrent_vector.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_sort.h"

namespace zinoviev_a_convex_hull_components_tbb {

struct Point {
  int x, y;

  bool operator<(const Point& other) const { return x < other.x || (x == other.x && y < other.y); }
  bool operator==(const Point& other) const = default;
};

class ConvexHullTBB : public ppc::core::Task {
 public:
  explicit ConvexHullTBB(ppc::core::TaskDataPtr task_data);
  bool PreProcessingImpl() noexcept override;
  bool ValidationImpl() noexcept override;
  bool RunImpl() noexcept override;
  bool PostProcessingImpl() noexcept override;

 private:
  std::vector<Point> input_points_;
  std::vector<Point> output_hull_;

  static std::vector<Point> FindConvexHull(const std::vector<Point>& points) noexcept;
  static int Cross(const Point& o, const Point& a, const Point& b) noexcept;
};

}  // namespace zinoviev_a_convex_hull_components_tbb