#include "omp/milovankin_m_histogram_stretching/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace milovankin_m_histogram_stretching_omp {

bool TestTaskOpenMP::ValidationImpl() {
  return !task_data->inputs.empty() && !task_data->inputs_count.empty() && task_data->inputs_count[0] != 0 &&
         !task_data->outputs.empty() && !task_data->outputs_count.empty() &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool TestTaskOpenMP::PreProcessingImpl() {
  const uint8_t* input_data = task_data->inputs.front();
  const uint32_t input_size = task_data->inputs_count.front();

  img_.assign(input_data, input_data + input_size);
  return true;
}

bool TestTaskOpenMP::RunImpl() {
  uint8_t min_val = std::numeric_limits<uint8_t>::max();
  uint8_t max_val = std::numeric_limits<uint8_t>::min();
  const size_t size = img_.size();

// Parallel min/max finding using reduction
#pragma omp parallel
  {
    uint8_t local_min = std::numeric_limits<uint8_t>::max();
    uint8_t local_max = std::numeric_limits<uint8_t>::min();

#pragma omp for nowait
    for (long long i = 0; i < static_cast<long long>(size); i++) {
      local_min = std::min(local_min, img_[i]);
      local_max = std::max(local_max, img_[i]);
    }

#pragma omp critical
    {
      min_val = std::min(min_val, local_min);
      max_val = std::max(max_val, local_max);
    }
  }

  if (min_val != max_val) {
    const int delta = max_val - min_val;

// Parallel pixel transformation
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(size); i++) {
      img_[i] = ((img_[i] - min_val) * 255 + delta / 2) / delta;
    }
  }

  return true;
}

bool TestTaskOpenMP::PostProcessingImpl() {
  uint8_t* output_data = task_data->outputs[0];
  const uint32_t output_size = task_data->outputs_count[0];
  const uint32_t copy_size = std::min(output_size, static_cast<uint32_t>(img_.size()));

  std::copy_n(img_.cbegin(), copy_size, output_data);
  return true;
}

}  // namespace milovankin_m_histogram_stretching_omp
