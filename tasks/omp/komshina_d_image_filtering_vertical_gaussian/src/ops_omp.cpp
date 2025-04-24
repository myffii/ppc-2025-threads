#include "omp/komshina_d_image_filtering_vertical_gaussian/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool komshina_d_image_filtering_vertical_gaussian_omp::TestTaskOpenMP::PreProcessingImpl() {
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

bool komshina_d_image_filtering_vertical_gaussian_omp::TestTaskOpenMP::ValidationImpl() {
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

bool komshina_d_image_filtering_vertical_gaussian_omp::TestTaskOpenMP::RunImpl() {
  int local_height = static_cast<int>(height_);
  int local_width = static_cast<int>(width_);
  std::vector<float> local_kernel = kernel_;

  auto &input_ref = input_;
  auto &output_ref = output_;

#pragma omp parallel default(none) shared(input_ref, output_ref) firstprivate(local_height, local_width, local_kernel)
  {
#pragma omp for
    for (int y = 1; y < local_height - 1; ++y) {
      for (int x = 1; x < local_width - 1; ++x) {
        std::size_t base_idx = (y * local_width + x) * 3;

        for (int c = 0; c < 3; ++c) {
          float total = 0.0F;
          int k_idx = 0;

          for (int ky = -1; ky <= 1; ++ky) {
            std::size_t row_idx = ((((y + ky) * local_width) + (x - 1)) * 3) + c;

            for (int kx = -1; kx <= 1; ++kx, ++k_idx) {
              total += static_cast<float>(input_ref[row_idx]) * local_kernel[k_idx];
              row_idx += 3;
            }
          }
          output_ref[base_idx + c] = std::clamp(static_cast<int>(std::round(total)), 0, 255);
        }
      }
    }
  }
  return true;
}

bool komshina_d_image_filtering_vertical_gaussian_omp::TestTaskOpenMP::PostProcessingImpl() {
  if (task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }

  auto *output_ptr = task_data->outputs[0];
  std::ranges::copy(output_, output_ptr);

  return true;
}
