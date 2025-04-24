#include "omp/sozonov_i_image_filtering_block_partitioning/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool sozonov_i_image_filtering_block_partitioning_omp::TestTaskOpenMP::PreProcessingImpl() {
  // Init image
  image_ = std::vector<double>(task_data->inputs_count[0]);
  auto *image_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  std::ranges::copy(image_ptr, image_ptr + task_data->inputs_count[0], image_.begin());

  width_ = static_cast<int>(task_data->inputs_count[1]);
  height_ = static_cast<int>(task_data->inputs_count[2]);

  // Init filtered image
  filtered_image_ = std::vector<double>(width_ * height_, 0);
  return true;
}

bool sozonov_i_image_filtering_block_partitioning_omp::TestTaskOpenMP::ValidationImpl() {
  // Init image
  image_ = std::vector<double>(task_data->inputs_count[0]);
  auto *image_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  std::ranges::copy(image_ptr, image_ptr + task_data->inputs_count[0], image_.begin());

  size_t img_size = task_data->inputs_count[1] * task_data->inputs_count[2];

  // Check pixels range from 0 to 255
  for (size_t i = 0; i < img_size; ++i) {
    if (image_[i] < 0 || image_[i] > 255) {
      return false;
    }
  }

  // Check size of image
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == img_size &&
         task_data->outputs_count[0] == img_size && task_data->inputs_count[1] >= 3 && task_data->inputs_count[2] >= 3;
}

bool sozonov_i_image_filtering_block_partitioning_omp::TestTaskOpenMP::RunImpl() {
  std::vector<double> kernel = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};

  int block_size = 32;

  int num_blocks_x = (width_ + block_size - 1) / block_size;
  int num_blocks_y = (height_ + block_size - 1) / block_size;

#pragma omp parallel for schedule(dynamic, 1)
  for (int block_y = 0; block_y < num_blocks_y; ++block_y) {
    for (int block_x = 0; block_x < num_blocks_x; ++block_x) {
      int start_i = block_y * block_size;
      int start_j = block_x * block_size;
      int end_i = std::min(start_i + block_size, height_ - 1);
      int end_j = std::min(start_j + block_size, width_ - 1);

      for (int i = std::max(1, start_i); i < end_i; ++i) {
        for (int j = std::max(1, start_j); j < end_j; ++j) {
          double sum = 0;
          for (int l = -1; l <= 1; ++l) {
            for (int k = -1; k <= 1; ++k) {
              sum += image_[((i - l) * width_) + (j - k)] * kernel[((l + 1) * 3) + (k + 1)];
            }
          }
          filtered_image_[(i * width_) + j] = sum;
        }
      }
    }
  }
  return true;
}

bool sozonov_i_image_filtering_block_partitioning_omp::TestTaskOpenMP::PostProcessingImpl() {
  auto *out = reinterpret_cast<double *>(task_data->outputs[0]);
  std::ranges::copy(filtered_image_.begin(), filtered_image_.end(), out);
  return true;
}
