#include "stl/ermolaev_v_graham_scan/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

int ermolaev_v_graham_scan_stl::TestTaskSTL::CrossProduct(const Point &p1, const Point &p2, const Point &p3) {
  return ((p2.x - p1.x) * (p3.y - p1.y)) - ((p3.x - p1.x) * (p2.y - p1.y));
}

bool ermolaev_v_graham_scan_stl::TestTaskSTL::IsAllCollinear() {
  if (input_.size() <= 2) {
    return true;
  }

  Point p1 = input_[0];
  Point p2 = input_[1];

  if (any_of(input_.begin() + 2, input_.end(), [&](const Point &p) { return CrossProduct(p1, p2, p) != 0; })) {
    return false;
  }

  int num_threads = ppc::util::GetPPCNumThreads();
  int input_size = static_cast<int>(input_.size());
  int chunk_size = input_size / num_threads;
  bool found_non_collinear = false;

  std::vector<std::thread> threads;

  auto worker = [&](int start, int end) {
    bool local_found = false;
    for (int i = start; i < end && !local_found; i++) {
      for (int j = i + 1; j < input_size && !local_found; j++) {
        for (int k = j + 1; k < input_size && !local_found; k++) {
          if (CrossProduct(input_[i], input_[j], input_[k]) != 0) {
            found_non_collinear = true;
            return;
          }
        }
      }
    }
  };

  int start = 0;
  int end = 0;
  for (int i = 0; i < num_threads; i++) {
    start = i * chunk_size;
    end = (i == num_threads - 1) ? input_size : start + chunk_size;
    threads.emplace_back(worker, start, end);
  }

  for (auto &t : threads) {
    t.join();
  }

  return !found_non_collinear;
}

bool ermolaev_v_graham_scan_stl::TestTaskSTL::IsAllSame() {
  if (input_.size() <= 1) {
    return true;
  }

  const Point &first = input_[0];
  return !any_of(input_.begin() + 1, input_.end(), [&](const Point &p) { return p != first; });
}

bool ermolaev_v_graham_scan_stl::TestTaskSTL::CheckGrahamNecessaryConditions() {
  if (input_.size() < kMinInputPoints) {
    return false;
  }

  return !IsAllSame() && !IsAllCollinear();
}

void ermolaev_v_graham_scan_stl::TestTaskSTL::GrahamScan() {
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

bool ermolaev_v_graham_scan_stl::TestTaskSTL::PreProcessingImpl() {
  auto *in_ptr = reinterpret_cast<Point *>(task_data->inputs[0]);
  input_ = std::vector<Point>(in_ptr, in_ptr + task_data->inputs_count[0]);
  output_ = std::vector<Point>();
  return true;
}

bool ermolaev_v_graham_scan_stl::TestTaskSTL::ValidationImpl() {
  return task_data->inputs_count[0] >= kMinInputPoints && task_data->inputs_count[0] <= task_data->outputs_count[0];
}

bool ermolaev_v_graham_scan_stl::TestTaskSTL::RunImpl() {
  if (!CheckGrahamNecessaryConditions()) {
    return false;
  }

  auto base_it = std::ranges::min_element(input_);
  std::iter_swap(input_.begin(), base_it);

  ParallelSort(input_.begin() + 1, input_.end(), [&](const Point &a, const Point &b) {
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

bool ermolaev_v_graham_scan_stl::TestTaskSTL::PostProcessingImpl() {
  task_data->outputs_count.clear();
  task_data->outputs_count.push_back(output_.size());
  std::ranges::copy(output_, reinterpret_cast<Point *>(task_data->outputs[0]));
  return true;
}