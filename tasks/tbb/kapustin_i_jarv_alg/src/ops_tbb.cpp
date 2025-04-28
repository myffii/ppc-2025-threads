#include "tbb/kapustin_i_jarv_alg/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

int kapustin_i_jarv_alg_tbb::TestTaskTBB::CalculateDistance(const std::pair<int, int>& p1,
                                                            const std::pair<int, int>& p2) {
  return static_cast<int>(std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2));
}

int kapustin_i_jarv_alg_tbb::TestTaskTBB::Orientation(const std::pair<int, int>& p, const std::pair<int, int>& q,
                                                      const std::pair<int, int>& r) {
  int val = ((q.second - p.second) * (r.first - q.first)) - ((q.first - p.first) * (r.second - q.second));
  if (val == 0) {
    return 0;
  }
  return (val > 0) ? 1 : -1;
}

bool kapustin_i_jarv_alg_tbb::TestTaskTBB::PreProcessingImpl() {
  std::vector<std::pair<int, int>> points;

  for (size_t i = 0; i < task_data->inputs.size(); ++i) {
    auto* data = reinterpret_cast<std::pair<int, int>*>(task_data->inputs[i]);
    size_t count = task_data->inputs_count[i];
    points.assign(data, data + count);
  }
  input_ = points;

  leftmost_index_ = 0;
  for (size_t i = 1; i < input_.size(); ++i) {
    if (input_[i].first < input_[leftmost_index_].first) {
      leftmost_index_ = i;
    }
  }

  current_point_ = input_[leftmost_index_];

  return true;
}

bool kapustin_i_jarv_alg_tbb::TestTaskTBB::ValidationImpl() { return !task_data->inputs.empty(); }

bool kapustin_i_jarv_alg_tbb::TestTaskTBB::RunImpl() {
  std::pair<int, int> start_point = current_point_;
  size_t current_index = leftmost_index_;
  output_.clear();
  output_.push_back(start_point);

  do {
    size_t next_index = (current_index + 1) % input_.size();

    size_t best_index = oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<size_t>(0, input_.size()), next_index,
        [&](const oneapi::tbb::blocked_range<size_t>& r, size_t local_best) -> size_t {
          for (size_t i = r.begin(); i != r.end(); ++i) {
            if (i == current_index) {
              continue;
            }

            int orientation = Orientation(input_[current_index], input_[local_best], input_[i]);
            if (orientation > 0 ||
                (orientation == 0 && CalculateDistance(input_[i], input_[current_index]) >
                                         CalculateDistance(input_[local_best], input_[current_index]))) {
              local_best = i;
            }
          }
          return local_best;
        },
        [&](size_t a, size_t b) -> size_t {
          int orientation = Orientation(input_[current_index], input_[a], input_[b]);
          if (orientation > 0 || (orientation == 0 && CalculateDistance(input_[b], input_[current_index]) >
                                                          CalculateDistance(input_[a], input_[current_index]))) {
            return b;
          }
          return a;
        });

    if (!output_.empty() && input_[best_index] == output_.front()) {
      break;
    }

    current_point_ = input_[best_index];
    output_.push_back(current_point_);
    current_index = best_index;

  } while (current_point_ != start_point);

  return true;
}

bool kapustin_i_jarv_alg_tbb::TestTaskTBB::PostProcessingImpl() {
  auto* result_ptr = reinterpret_cast<std::pair<int, int>*>(task_data->outputs[0]);
  std::ranges::copy(output_, result_ptr);
  return true;
}
