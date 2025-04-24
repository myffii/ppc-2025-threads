#pragma once

#include <tbb/tbb.h>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shulpin_i_jarvis_tbb {

struct Point {
  double x, y;
  Point() : x(0), y(0) {}
  Point(double x_coordinate, double y_coordinate) : x(x_coordinate), y(y_coordinate) {}

  bool operator==(const Point& other) const { return x == other.x && y == other.y; }
};

struct PointHash {
  size_t operator()(const shulpin_i_jarvis_tbb::Point& p) const {
    size_t hx = std::hash<double>{}(p.x);
    size_t hy = std::hash<double>{}(p.y);
    return hx ^ (hy << 1);
  }
};

class JarvisSequential : public ppc::core::Task {
 public:
  explicit JarvisSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void MakeJarvisPassage(std::vector<shulpin_i_jarvis_tbb::Point>& input,
                                std::vector<shulpin_i_jarvis_tbb::Point>& output);

 private:
  std::vector<shulpin_i_jarvis_tbb::Point> input_seq_, output_seq_;
};

class JarvisTBBParallel : public ppc::core::Task {
 public:
  explicit JarvisTBBParallel(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void MakeJarvisPassageTBB(std::vector<shulpin_i_jarvis_tbb::Point>& input,
                                   std::vector<shulpin_i_jarvis_tbb::Point>& output);

 private:
  std::vector<shulpin_i_jarvis_tbb::Point> input_tbb_, output_tbb_;
};

}  // namespace shulpin_i_jarvis_tbb