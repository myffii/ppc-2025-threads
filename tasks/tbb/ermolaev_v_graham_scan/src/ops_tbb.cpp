#include "tbb/ermolaev_v_graham_scan/include/ops_tbb.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_reduce.h"
#include "oneapi/tbb/parallel_sort.h"

int ermolaev_v_graham_scan_tbb::TestTaskTBB::CrossProduct(const Point &p1, const Point &p2, const Point &p3) {
  return ((p2.x - p1.x) * (p3.y - p1.y)) - ((p3.x - p1.x) * (p2.y - p1.y));
}

size_t ermolaev_v_graham_scan_tbb::TestTaskTBB::IndexOfMinElement() {
  int input_size = static_cast<int>(input_.size());
  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, input_size), 0,
      [&](const tbb::blocked_range<int> &range, int init) -> int {
        for (int i = range.begin(); i != range.end(); i++) {
          if (input_[i] < input_[init]) {
            init = i;
          }
        }
        return init;
      },
      [&](int a, int b) -> int { return input_[a] < input_[b] ? a : b; });
}

bool ermolaev_v_graham_scan_tbb::TestTaskTBB::IsAllCollinear() {
  int input_size = static_cast<int>(input_.size());

  auto is_non_collinear = tbb::parallel_reduce(
      tbb::blocked_range<int>(0, input_size - 2), false,
      [&](const tbb::blocked_range<int> &range, bool init) -> bool {
        bool found_non_collinear = init;
        if (!found_non_collinear) {
          for (int i = range.begin(); i != range.end() && !found_non_collinear; ++i) {
            for (int j = i + 1; j < input_size - 1 && !found_non_collinear; ++j) {
              for (int k = j + 1; k < input_size && !found_non_collinear; ++k) {
                if (CrossProduct(input_[i], input_[j], input_[k]) != 0) {
                  found_non_collinear = true;
                }
              }
            }
          }
        }
        return found_non_collinear;
      },
      [](bool a, bool b) -> bool { return a || b; });

  return !is_non_collinear;
}

bool ermolaev_v_graham_scan_tbb::TestTaskTBB::IsAllSame() {
  const Point &first = input_[0];
  int input_size = static_cast<int>(input_.size());

  auto all_same = tbb::parallel_reduce(
      tbb::blocked_range<int>(1, input_size), true,
      [&](const tbb::blocked_range<int> &range, bool init) -> bool {
        return init && std::ranges::all_of(input_.begin() + range.begin(), input_.begin() + range.end(),
                                           [&first](const Point &p) { return p == first; });
      },
      [](bool a, bool b) -> bool { return a && b; });

  return all_same;
}

bool ermolaev_v_graham_scan_tbb::TestTaskTBB::CheckGrahamNecessaryConditions() {
  if (input_.size() < kMinInputPoints) {
    return false;
  }

  return !IsAllSame() && !IsAllCollinear();
}

void ermolaev_v_graham_scan_tbb::TestTaskTBB::GrahamScan() {
  output_.clear();
  output_.emplace_back(input_[0]);
  output_.emplace_back(input_[1]);

  Point p1;
  Point p2;
  Point p3;
  for (size_t i = kMinStackPoints; i < input_.size(); i++) {
    while (output_.size() >= kMinStackPoints) {
      p1 = output_[output_.size() - 2];
      p2 = output_[output_.size() - 1];
      p3 = input_[i];

      int cross = CrossProduct(p1, p2, p3);

      if (cross > 0) {
        break;
      }
      output_.pop_back();
    }
    output_.emplace_back(input_[i]);
  }
}

bool ermolaev_v_graham_scan_tbb::TestTaskTBB::PreProcessingImpl() {
  auto *in_ptr = reinterpret_cast<Point *>(task_data->inputs[0]);
  input_ = std::vector<Point>(in_ptr, in_ptr + task_data->inputs_count[0]);
  output_ = std::vector<Point>();
  return true;
}

bool ermolaev_v_graham_scan_tbb::TestTaskTBB::ValidationImpl() {
  return task_data->inputs_count[0] >= kMinInputPoints && task_data->inputs_count[0] <= task_data->outputs_count[0];
}

bool ermolaev_v_graham_scan_tbb::TestTaskTBB::RunImpl() {
  if (!CheckGrahamNecessaryConditions()) {
    return false;
  }

  size_t min_idx = IndexOfMinElement();
  std::iter_swap(input_.begin(), input_.begin() + static_cast<int>(min_idx));

  tbb::parallel_sort(input_.begin() + 1, input_.end(), [&](const Point &a, const Point &b) {
    auto squared_dist = [](const Point &p1, const Point &p2) -> int {
      int dx = p1.x - p2.x;
      int dy = p1.y - p2.y;
      return ((dx * dx) + (dy * dy));
    };

    int cross = CrossProduct(input_[0], a, b);
    if (cross == 0) {
      return squared_dist(a, input_[0]) < squared_dist(b, input_[0]);
    }

    return cross > 0;
  });

  GrahamScan();

  return true;
}

bool ermolaev_v_graham_scan_tbb::TestTaskTBB::PostProcessingImpl() {
  task_data->outputs_count.clear();
  task_data->outputs_count.push_back(output_.size());
  std::ranges::copy(output_, reinterpret_cast<Point *>(task_data->outputs[0]));
  return true;
}