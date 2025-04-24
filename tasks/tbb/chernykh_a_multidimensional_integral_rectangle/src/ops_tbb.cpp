#include "tbb/chernykh_a_multidimensional_integral_rectangle/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

namespace chernykh_a_multidimensional_integral_rectangle_tbb {

double Dimension::GetLowerBound() const { return lower_bound_; }

double Dimension::GetUpperBound() const { return upper_bound_; }

int Dimension::GetStepsCount() const { return steps_count_; }

double Dimension::GetStepSize() const { return (upper_bound_ - lower_bound_) / steps_count_; }

bool Dimension::IsValid() const { return lower_bound_ < upper_bound_ && steps_count_ > 0; }

bool TBBTask::ValidationImpl() {
  auto *dims_ptr = reinterpret_cast<Dimension *>(task_data->inputs[0]);
  uint32_t dims_size = task_data->inputs_count[0];
  return dims_size > 0 &&
         std::all_of(dims_ptr, dims_ptr + dims_size, [](const Dimension &dim) -> bool { return dim.IsValid(); });
}

bool TBBTask::PreProcessingImpl() {
  auto *dims_ptr = reinterpret_cast<Dimension *>(task_data->inputs[0]);
  uint32_t dims_size = task_data->inputs_count[0];
  dims_.assign(dims_ptr, dims_ptr + dims_size);
  return true;
}

bool TBBTask::RunImpl() {
  int total_points = GetTotalPoints();
  result_ = oneapi::tbb::parallel_reduce(
      oneapi::tbb::blocked_range(0, total_points), 0.0,
      [&](const oneapi::tbb::blocked_range<int> &range, double accum) -> double {
        auto point = Point(dims_.size());
        for (int i = range.begin(); i < range.end(); i++) {
          FillPoint(i, point);
          accum += func_(point);
        }
        return accum;
      },
      std::plus());
  result_ *= GetScalingFactor();
  return true;
}

bool TBBTask::PostProcessingImpl() {
  *reinterpret_cast<double *>(task_data->outputs[0]) = result_;
  return true;
}

void TBBTask::FillPoint(int index, Point &point) const {
  for (size_t i = 0; i < dims_.size(); i++) {
    int coordinate_index = index % dims_[i].GetStepsCount();
    point[i] = dims_[i].GetLowerBound() + (coordinate_index + 1) * dims_[i].GetStepSize();
    index /= dims_[i].GetStepsCount();
  }
}

int TBBTask::GetTotalPoints() const {
  return std::accumulate(dims_.begin(), dims_.end(), 1,
                         [](int accum, const Dimension &dim) -> int { return accum * dim.GetStepsCount(); });
}

double TBBTask::GetScalingFactor() const {
  return std::accumulate(dims_.begin(), dims_.end(), 1.0,
                         [](double accum, const Dimension &dim) -> double { return accum * dim.GetStepSize(); });
}

}  // namespace chernykh_a_multidimensional_integral_rectangle_tbb
