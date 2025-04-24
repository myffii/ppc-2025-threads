#include "tbb/titov_s_ImageFilter_HorizGaussian3x3/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "oneapi/tbb/parallel_for.h"

bool titov_s_image_filter_horiz_gaussian3x3_tbb::ImageFilterTBB::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  width_ = height_ = static_cast<int>(std::sqrt(input_size));
  input_.assign(in_ptr, in_ptr + input_size);

  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  output_ = std::vector<double>(input_size, 0.0);

  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_tbb::ImageFilterTBB::ValidationImpl() {
  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  size_t size = input_.size();
  auto sqrt_size = static_cast<size_t>(std::sqrt(size));
  if (kernel_.size() != 3 || sqrt_size * sqrt_size != size) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool titov_s_image_filter_horiz_gaussian3x3_tbb::ImageFilterTBB::RunImpl() {
  const double k0 = kernel_[0];
  const double k1 = kernel_[1];
  const double k2 = kernel_[2];
  const double inv_sum = 1.0 / (k0 + k1 + k2);

  const int width = width_;
  const int height = height_;
  const double *input = input_.data();
  double *output = output_.data();

  tbb::parallel_for(
      tbb::blocked_range<int>(0, height),
      [&](const tbb::blocked_range<int> &range) {
        for (int row = range.begin(); row < range.end(); ++row) {
          const double *row_in = input + (row * width);
          double *row_out = output + (row * width);

          row_out[0] = (0.0 * k0 + row_in[0] * k1 + row_in[1] * k2) * inv_sum;

          for (int col = 1; col < width - 1; ++col) {
            row_out[col] = (row_in[col - 1] * k0 + row_in[col] * k1 + row_in[col + 1] * k2) * inv_sum;
          }

          if (width > 1) {
            row_out[width - 1] = (row_in[width - 2] * k0 + row_in[width - 1] * k1 + 0.0 * k2) * inv_sum;
          }
        }
      },
      tbb::auto_partitioner());

  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_tbb::ImageFilterTBB::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);

  for (size_t i = 0; i < output_.size(); i++) {
    out_ptr[i] = output_[i];
  }

  return true;
}
