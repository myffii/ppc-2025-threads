#pragma once

#include <oneapi/tbb/blocked_range.h>
#include <tbb/tbb.h>

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace oturin_a_gift_wrapping_tbb {

struct Coord {
  int x, y;
  bool operator==(const Coord o) const { return (x == o.x && y == o.y); }
  bool operator!=(const Coord o) const { return x != o.x || y != o.y; }
};

double Distance(Coord a, Coord b);

// Angle Between Three Points
double ABTP(Coord a, Coord b, Coord c);

// Angle Between Three Points for leftmost point
double ABTP(Coord a, Coord c);

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  class PointSearcher;

 private:
  std::vector<Coord> input_, output_;
  int n_;

  int FindMostLeft();
  void PointSearch(double t, double &line_angle, int &search_index, int i);
};

class TestTaskTBB::PointSearcher {
  TestTaskTBB *p_;
  double line_angle_ = -5;

 public:
  int search_index;

  PointSearcher(TestTaskTBB *parent, int search_index) : p_(parent), search_index(search_index) {}

  PointSearcher(PointSearcher &x, tbb::split) : p_(x.p_), search_index(x.search_index) {}

  void operator()(const tbb::blocked_range<int> &r);

  void join(const PointSearcher &x);  // NOLINT(readability-identifier-naming): tbb using this name
};

}  // namespace oturin_a_gift_wrapping_tbb