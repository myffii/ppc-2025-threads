#include "tbb/varfolomeev_g_histogram_linear_stretching/include/ops_tbb.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "oneapi/tbb/parallel_for.h"

bool varfolomeev_g_histogram_linear_stretching_tbb::TestTaskTBB ::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<uint8_t *>(task_data->inputs[0]);
  img_ = std::vector<uint8_t>(in_ptr, in_ptr + input_size);
  res_.resize(img_.size());
  return true;
}

bool varfolomeev_g_histogram_linear_stretching_tbb::TestTaskTBB ::ValidationImpl() {
  return task_data->outputs_count[0] > 0 && task_data->inputs_count[0] > 0 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool varfolomeev_g_histogram_linear_stretching_tbb::TestTaskTBB ::RunImpl() {
  uint8_t min_val = *std::ranges::min_element(img_);
  uint8_t max_val = *std::ranges::max_element(img_);

  if (max_val != min_val) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, img_.size()), [&](const tbb::blocked_range<size_t> &r) {
      for (size_t i = r.begin(); i != r.end(); ++i) {
        res_[i] = static_cast<uint8_t>(std::round(static_cast<double>(img_[i] - min_val) * 255 / (max_val - min_val)));
      }
    });
  } else {
    std::ranges::fill(res_.begin(), res_.end(), min_val);
  }
  return true;
}

bool varfolomeev_g_histogram_linear_stretching_tbb::TestTaskTBB ::PostProcessingImpl() {
  std::memcpy(task_data->outputs[0], res_.data(), res_.size() * sizeof(uint8_t));

  return true;
}