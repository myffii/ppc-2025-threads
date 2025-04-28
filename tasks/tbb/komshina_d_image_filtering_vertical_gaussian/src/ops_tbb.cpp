#include "tbb/komshina_d_image_filtering_vertical_gaussian/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <vector>

#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

namespace {
void GaussianVerticalFilter(const std::vector<unsigned char> &in_vec, std::size_t width, std::size_t height,
                            const std::vector<float> &kernel, std::vector<unsigned char> &out_vec) {
  constexpr int kChannels = 3;
  const int k_radius = 1;

  for (std::size_t y = 1; y + 1 < height; ++y) {
    for (std::size_t x = 1; x + 1 < width; ++x) {
      std::size_t base_idx = (y * width + x) * kChannels;

      for (std::size_t c = 0; c < kChannels; ++c) {
        float total = 0.0F;
        std::size_t k_idx = 0;

        for (int ky = -k_radius; ky <= k_radius; ++ky) {
          std::size_t row_idx = ((((y + ky) * width) + (x - 1)) * kChannels) + c;

          for (int kx = -k_radius; kx <= k_radius; ++kx, ++k_idx) {
            total += static_cast<float>(in_vec[row_idx]) * kernel[k_idx];
            row_idx += kChannels;
          }
        }

        out_vec[base_idx + c] = std::clamp(static_cast<int>(std::round(total)), 0, 255);
      }
    }
  }
}
}  // namespace

bool komshina_d_image_filtering_vertical_gaussian_tbb::TestTaskTBB::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];

  unsigned int input_size = width_ * height_ * 3;
  auto *in_ptr = reinterpret_cast<unsigned char *>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);

  unsigned int kernel_size = task_data->inputs_count[2];
  auto *kernel_ptr = reinterpret_cast<float *>(task_data->inputs[1]);
  kernel_.assign(kernel_ptr, kernel_ptr + kernel_size);

  output_.assign(input_.begin(), input_.end());

  return true;
}

bool komshina_d_image_filtering_vertical_gaussian_tbb::TestTaskTBB::ValidationImpl() {
  if (task_data->inputs[0] == nullptr || task_data->inputs[1] == nullptr || task_data->outputs.empty() ||
      task_data->outputs[0] == nullptr) {
    return false;
  }

  const auto &input_count = task_data->inputs_count;
  const auto &output_count = task_data->outputs_count;

  if (input_count.size() < 3 || output_count.empty()) {
    return false;
  }

  constexpr int kKernelSize = 9;
  constexpr int kChannels = 3;

  bool valid_kernel = (input_count[2] == kKernelSize);
  bool valid_size = (input_count[0] * input_count[1] * kChannels == output_count[0]);

  return valid_kernel && valid_size;
}

bool komshina_d_image_filtering_vertical_gaussian_tbb::TestTaskTBB::RunImpl() {
  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  arena.execute([&] {
    oneapi::tbb::task_group tg;
    for (int thr = 0; thr < ppc::util::GetPPCNumThreads(); ++thr) {
      tg.run([&] { GaussianVerticalFilter(input_, width_, height_, kernel_, output_); });
    }
    tg.wait();
  });

  return true;
}

bool komshina_d_image_filtering_vertical_gaussian_tbb::TestTaskTBB::PostProcessingImpl() {
  if (task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }

  auto *output_ptr = task_data->outputs[0];
  std::ranges::copy(output_, output_ptr);

  return true;
}