#include "stl/varfolomeev_g_histogram_linear_stretching/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool varfolomeev_g_histogram_linear_stretching_stl::TestTaskSTL ::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<uint8_t *>(task_data->inputs[0]);
  img_ = std::vector<uint8_t>(in_ptr, in_ptr + input_size);
  res_.resize(img_.size());
  return true;
}

bool varfolomeev_g_histogram_linear_stretching_stl::TestTaskSTL ::ValidationImpl() {
  return task_data->outputs_count[0] > 0 && task_data->inputs_count[0] > 0 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool varfolomeev_g_histogram_linear_stretching_stl::TestTaskSTL ::RunImpl() {
  uint8_t min_val = *std::ranges::min_element(img_);
  uint8_t max_val = *std::ranges::max_element(img_);

  if (max_val != min_val) {
    const size_t num_threads = ppc::util::GetPPCNumThreads();
    std::vector<std::thread> threads;
    const size_t chunk_size = img_.size() / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
      const size_t start = t * chunk_size;
      const size_t end = (t == num_threads - 1) ? img_.size() : start + chunk_size;

      threads.emplace_back([this, start, end, min_val, max_val]() {
        for (size_t i = start; i < end; ++i) {
          res_[i] =
              static_cast<uint8_t>(std::round(static_cast<double>(img_[i] - min_val) * 255 / (max_val - min_val)));
        }
      });
    }
    for (auto &thread : threads) {
      thread.join();
    }
  } else {
    std::ranges::fill(res_.begin(), res_.end(), min_val);
  }
  return true;
}

bool varfolomeev_g_histogram_linear_stretching_stl::TestTaskSTL ::PostProcessingImpl() {
  std::memcpy(task_data->outputs[0], res_.data(), res_.size() * sizeof(uint8_t));

  return true;
}
