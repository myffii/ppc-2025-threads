#include "tbb/kozlova_e_contrast_enhancement/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task_arena.h"

bool kozlova_e_contrast_enhancement_tbb::TestTaskTBB::PreProcessingImpl() {
  auto *input_ptr = reinterpret_cast<uint8_t *>(task_data->inputs[0]);
  size_t size = task_data->inputs_count[0];
  width_ = task_data->inputs_count[1];
  height_ = task_data->inputs_count[2];
  output_.resize(size, 0);
  input_.resize(size);
  std::copy(input_ptr, input_ptr + size, input_.begin());

  return true;
}

bool kozlova_e_contrast_enhancement_tbb::TestTaskTBB::ValidationImpl() {
  size_t size = task_data->inputs_count[0];
  size_t check_width = task_data->inputs_count[1];
  size_t check_height = task_data->inputs_count[2];
  return size == task_data->outputs_count[0] && size > 0 && (size % 2 == 0) && check_width >= 1 && check_height >= 1 &&
         (size == check_height * check_width);
}

bool kozlova_e_contrast_enhancement_tbb::TestTaskTBB::RunImpl() {
  uint8_t min_value = *std::ranges::min_element(input_);
  uint8_t max_value = *std::ranges::max_element(input_);

  if (min_value == max_value) {
    std::ranges::copy(input_, output_.begin());
    return true;
  }

  oneapi::tbb::task_arena task_arena(ppc::util::GetPPCNumThreads());
  task_arena.execute([&] {
    tbb::parallel_for(tbb::blocked_range<int>(0, (int)input_.size()), [&](const tbb::blocked_range<int> &r) {
      for (int i = r.begin(); i < r.end(); ++i) {
        output_[i] =
            static_cast<uint8_t>(((input_[i] - min_value) / static_cast<double>(max_value - min_value)) * 255.0);
        output_[i] = std::clamp(static_cast<int>(output_[i]), 0, 255);
      }
    });
  });

  return true;
}

bool kozlova_e_contrast_enhancement_tbb::TestTaskTBB::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<uint8_t *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
