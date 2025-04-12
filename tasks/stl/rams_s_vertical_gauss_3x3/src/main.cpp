#include "stl/rams_s_vertical_gauss_3x3/include/main.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool rams_s_vertical_gauss_3x3_stl::TaskStl::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];
  input_ = std::vector<uint8_t>(task_data->inputs[0], task_data->inputs[0] + (height_ * width_ * 3));
  auto *k = reinterpret_cast<float *>(task_data->inputs[1]);
  kernel_ = std::vector<float>(k, k + task_data->inputs_count[2]);

  output_ = std::vector<uint8_t>(input_);

  return true;
}

bool rams_s_vertical_gauss_3x3_stl::TaskStl::ValidationImpl() {
  return task_data->inputs_count[2] == 9 &&
         (task_data->inputs_count[0] * task_data->inputs_count[1] * 3) == task_data->outputs_count[0];
}

bool rams_s_vertical_gauss_3x3_stl::TaskStl::RunImpl() {
  if (height_ == 0 || width_ == 0) {
    return true;
  }
  const std::size_t num_threads = std::min(std::size_t(ppc::util::GetPPCNumThreads()), std::size_t(width_));
  std::vector<std::thread> threads(num_threads);
  for (std::size_t thread_i = 0; thread_i < num_threads; thread_i++) {
    threads[thread_i] = std::thread([&, thread_i] {
      std::size_t amount = (width_ / num_threads) + (thread_i < width_ % num_threads ? 1 : 0);
      std::size_t left = ((width_ / num_threads) * thread_i) + std::min(width_ % num_threads, thread_i);
      std::size_t right = std::min(left + amount, std::size_t(width_) - 1);
      for (std::size_t x = std::max(left, std::size_t(1)); x < right; x++) {
        for (std::size_t y = 1; y < height_ - 1; y++) {
          for (std::size_t i = 0; i < 3; i++) {
            output_[((y * width_ + x) * 3) + i] = std::clamp(static_cast<int>(std::round(
#define INNER(Y_SHIFT, X_SHIFT) \
  input_[((((y + (Y_SHIFT)) * width_) + x + (X_SHIFT)) * 3) + i] * kernel_[4 + (3 * (Y_SHIFT)) + (X_SHIFT)]
#define OUTER(Y) (INNER(Y, -1) + INNER(Y, 0) + INNER(Y, 1))
                                                                 (OUTER(-1) + OUTER(0) + OUTER(1))
#undef OUTER
#undef INNER
                                                                     )),
                                                             0, 255);
          }
        }
      }
    });
  }
  for (std::size_t thread_i = 0; thread_i < num_threads; thread_i++) {
    threads[thread_i].join();
  }
  return true;
}

bool rams_s_vertical_gauss_3x3_stl::TaskStl::PostProcessingImpl() {
  std::ranges::copy(output_, task_data->outputs[0]);
  return true;
}
