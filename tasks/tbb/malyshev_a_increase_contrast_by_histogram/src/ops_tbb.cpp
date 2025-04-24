#include "tbb/malyshev_a_increase_contrast_by_histogram/include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

bool malyshev_a_increase_contrast_by_histogram_tbb::TestTaskTBB::PreProcessingImpl() {
  data_.assign(task_data->inputs[0], task_data->inputs[0] + task_data->inputs_count[0]);

  return !data_.empty();
}

bool malyshev_a_increase_contrast_by_histogram_tbb::TestTaskTBB::ValidationImpl() {
  return task_data->inputs[0] != nullptr && task_data->outputs[0] != nullptr && task_data->inputs_count.size() == 1 &&
         task_data->outputs_count.size() == 1 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool malyshev_a_increase_contrast_by_histogram_tbb::TestTaskTBB::RunImpl() {
  size_t data_size = data_.size();
  size_t grain_size = data_size / ppc::util::GetPPCNumThreads();

  auto minmax = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, data_size, grain_size),
      std::make_pair(std::numeric_limits<uint8_t>::max(), std::numeric_limits<uint8_t>::min()),
      [&](const tbb::blocked_range<size_t>& range, std::pair<uint8_t, uint8_t> local_minmax) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
          uint8_t& val = data_[i];
          local_minmax.first = std::min(val, local_minmax.first);
          local_minmax.second = std::max(val, local_minmax.second);
        }
        return local_minmax;
      },
      [](std::pair<uint8_t, uint8_t> x, std::pair<uint8_t, uint8_t> y) {
        return std::make_pair(std::min(x.first, y.first), std::max(x.second, y.second));
      });

  uint8_t min_value = minmax.first;
  uint8_t max_value = minmax.second;

  if (min_value == max_value) {
    return true;
  }

  const auto spectrum = std::numeric_limits<uint8_t>::max();
  const auto range = max_value - min_value;

  tbb::parallel_for(tbb::blocked_range<size_t>(0, data_size, grain_size), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      data_[i] = static_cast<uint8_t>((data_[i] - min_value) * spectrum / range);
    }
  });

  return true;
}

bool malyshev_a_increase_contrast_by_histogram_tbb::TestTaskTBB::PostProcessingImpl() {
  std::ranges::copy(data_, task_data->outputs[0]);

  return true;
}
