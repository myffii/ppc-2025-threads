#include "stl/malyshev_a_increase_contrast_by_histogram/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool malyshev_a_increase_contrast_by_histogram_stl::TestTaskSTL::PreProcessingImpl() {
  data_.assign(task_data->inputs[0], task_data->inputs[0] + task_data->inputs_count[0]);
  return !data_.empty();
}

bool malyshev_a_increase_contrast_by_histogram_stl::TestTaskSTL::ValidationImpl() {
  return task_data->inputs[0] != nullptr && task_data->outputs[0] != nullptr && task_data->inputs_count.size() == 1 &&
         task_data->outputs_count.size() == 1 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool malyshev_a_increase_contrast_by_histogram_stl::TestTaskSTL::RunImpl() {
  int data_size = static_cast<int>(data_.size());
  int num_threads = ppc::util::GetPPCNumThreads();
  int grain_size = data_size / num_threads;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  uint8_t min_value = 0;
  uint8_t max_value = 0;

  auto [temp_min, temp_max] = std::ranges::minmax_element(data_);
  min_value = *temp_min;
  max_value = *temp_max;

  if (min_value == max_value) {
    return true;
  }

  const auto spectrum = std::numeric_limits<uint8_t>::max();
  const auto range = max_value - min_value;

  threads.clear();
  for (int i = 0; i < num_threads; ++i) {
    int start = i * grain_size;
    int end = (i == num_threads - 1) ? data_size : start + grain_size;
    threads.emplace_back([&, start, end]() {
      std::ranges::for_each(data_.begin() + start, data_.begin() + end, [&](uint8_t& value) {
        value = static_cast<uint8_t>((value - min_value) * spectrum / range);
      });
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return true;
}

bool malyshev_a_increase_contrast_by_histogram_stl::TestTaskSTL::PostProcessingImpl() {
  std::ranges::copy(data_, task_data->outputs[0]);
  return true;
}
