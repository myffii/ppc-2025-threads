#include "omp/kapustin_i_jarv_alg/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

int kapustin_i_jarv_alg_omp::TestTaskOMP::CalculateDistance(const std::pair<int, int>& p1,
                                                            const std::pair<int, int>& p2) {
  return static_cast<int>(std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2));
}
int kapustin_i_jarv_alg_omp::TestTaskOMP::Orientation(const std::pair<int, int>& p, const std::pair<int, int>& q,
                                                      const std::pair<int, int>& r) {
  int val = ((q.second - p.second) * (r.first - q.first)) - ((q.first - p.first) * (r.second - q.second));
  if (val == 0) {
    return 0;
  }
  return (val > 0) ? 1 : -1;
}

bool kapustin_i_jarv_alg_omp::TestTaskOMP::PreProcessingImpl() {
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

bool kapustin_i_jarv_alg_omp::TestTaskOMP::ValidationImpl() { return !task_data->inputs.empty(); }

bool kapustin_i_jarv_alg_omp::TestTaskOMP::RunImpl() {
  std::pair<int, int> start_point = current_point_;
  size_t current_index = leftmost_index_;
  output_.clear();
  output_.push_back(start_point);

  do {
    size_t next_index = (current_index + 1) % input_.size();

#pragma omp parallel
    {
      int local_next = static_cast<int>(next_index);

#pragma omp for nowait
      for (int i = 0; i < static_cast<int>(input_.size()); ++i) {
        if (static_cast<size_t>(i) == current_index) {
          continue;
        }

        int orientation = Orientation(input_[current_index], input_[local_next], input_[i]);

        if (orientation == 0) {
          int dist_next = CalculateDistance(input_[local_next], input_[current_index]);
          int dist_i = CalculateDistance(input_[i], input_[current_index]);
          if (dist_i > dist_next) {
            local_next = i;
          }
        } else if (orientation > 0) {
          local_next = i;
        }
      }

#pragma omp critical
      {
        int orientation = Orientation(input_[current_index], input_[next_index], input_[local_next]);
        if (orientation > 0) {
          next_index = static_cast<size_t>(local_next);
        } else if (orientation == 0) {
          int dist_next = CalculateDistance(input_[next_index], input_[current_index]);
          int dist_local = CalculateDistance(input_[local_next], input_[current_index]);
          if (dist_local > dist_next) {
            next_index = static_cast<size_t>(local_next);
          }
        }
      }
    }

    if (!output_.empty() && input_[next_index] == output_.front()) {
      break;
    }

    current_point_ = input_[next_index];
    output_.push_back(current_point_);
    current_index = next_index;

  } while (current_point_ != start_point);

  return true;
}

bool kapustin_i_jarv_alg_omp::TestTaskOMP::PostProcessingImpl() {
  auto* result_ptr = reinterpret_cast<std::pair<int, int>*>(task_data->outputs[0]);
  std::ranges::copy(output_, result_ptr);
  return true;
}