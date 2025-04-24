#include "../include/ops_tbb.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/parallel_reduce.h"

namespace milovankin_m_histogram_stretching_tbb {

bool TestTaskParallel::ValidationImpl() {
  return !task_data->inputs.empty() && !task_data->inputs_count.empty() && task_data->inputs_count[0] != 0 &&
         !task_data->outputs.empty() && !task_data->outputs_count.empty() &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool TestTaskParallel::PreProcessingImpl() {
  const uint8_t* input_data = task_data->inputs.front();
  const uint32_t input_size = task_data->inputs_count.front();

  img_.assign(input_data, input_data + input_size);
  return true;
}

struct MinMaxPair {
  uint8_t min_val;
  uint8_t max_val;

  MinMaxPair() : min_val(std::numeric_limits<uint8_t>::max()), max_val(0) {}

  MinMaxPair(uint8_t min, uint8_t max) : min_val(min), max_val(max) {}
};

bool TestTaskParallel::RunImpl() {
  const std::size_t grain_size = std::max(std::size_t(1024), img_.size() / 16);

  const std::vector<uint8_t>& img_ref = img_;

  MinMaxPair minmax = tbb::parallel_reduce(
      tbb::blocked_range<std::size_t>(0, img_.size(), grain_size), MinMaxPair(),
      [&img_ref](const tbb::blocked_range<std::size_t>& range, MinMaxPair init) -> MinMaxPair {
        uint8_t local_min = init.min_val;
        uint8_t local_max = init.max_val;

        for (std::size_t i = range.begin(); i != range.end(); ++i) {
          uint8_t val = img_ref[i];
          local_min = std::min(val, local_min);
          local_max = std::max(val, local_max);
        }

        return {local_min, local_max};
      },
      [](MinMaxPair a, MinMaxPair b) -> MinMaxPair {
        return {std::min(a.min_val, b.min_val), std::max(a.max_val, b.max_val)};
      });

  uint8_t min_val = minmax.min_val;
  uint8_t max_val = minmax.max_val;

  if (min_val != max_val) {
    const int delta = max_val - min_val;

    // Create a mutable reference to img_ that can be used in the lambda
    std::vector<uint8_t>& img_ref_mut = img_;

    // Apply stretching in parallel using a lambda directly
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, img_.size(), grain_size),
                      [min_val, delta, &img_ref_mut](const tbb::blocked_range<std::size_t>& range) {
                        for (std::size_t i = range.begin(); i != range.end(); ++i) {
                          img_ref_mut[i] = static_cast<uint8_t>(((img_ref_mut[i] - min_val) * 255 + delta / 2) / delta);
                        }
                      });
  }

  return true;
}

bool TestTaskParallel::PostProcessingImpl() {
  uint8_t* output_data = task_data->outputs[0];
  const uint32_t output_size = task_data->outputs_count[0];
  const uint32_t copy_size = std::min(output_size, static_cast<uint32_t>(img_.size()));

  std::copy_n(img_.cbegin(), copy_size, output_data);
  return true;
}

}  // namespace milovankin_m_histogram_stretching_tbb
