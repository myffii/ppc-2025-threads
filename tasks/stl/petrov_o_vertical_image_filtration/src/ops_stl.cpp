#include "stl/petrov_o_vertical_image_filtration/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool petrov_o_vertical_image_filtration_stl::TaskSTL::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];
  size_t input_size = width_ * height_;

  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  output_ = std::vector<int>((width_ - 2) * (height_ - 2), 0);
  return true;
}

bool petrov_o_vertical_image_filtration_stl::TaskSTL::ValidationImpl() {
  size_t width = task_data->inputs_count[0];
  size_t height = task_data->inputs_count[1];

  if (width < 3 || height < 3) {  // Check if the input image is too small
    return false;
  }

  return task_data->outputs_count[0] == (width - 2) * (height - 2);
}

// Thread's func
void petrov_o_vertical_image_filtration_stl::TaskSTL::ProcessImageStripe(size_t start_row, size_t end_row) {
  for (size_t i = start_row; i < end_row; ++i) {
    for (size_t j = 1; j < width_ - 1; ++j) {
      float sum = 0.0F;
      for (int ki = -1; ki <= 1; ++ki) {
        for (int kj = -1; kj <= 1; ++kj) {
          sum +=
              static_cast<float>(input_[((i + ki) * width_) + (j + kj)]) * gaussian_kernel_[((ki + 1) * 3) + (kj + 1)];
        }
      }
      output_[((i - 1) * (width_ - 2)) + (j - 1)] = static_cast<int>(sum);
    }
  }
}

bool petrov_o_vertical_image_filtration_stl::TaskSTL::RunImpl() {
  unsigned int num_threads = ppc::util::GetPPCNumThreads();

  num_threads = std::min(num_threads, static_cast<unsigned int>(height_ - 2));

  if (num_threads <= 1) {
    ProcessImageStripe(1, height_ - 1);
    return true;
  }

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  size_t rows_per_thread = (height_ - 2) / num_threads;
  size_t remainder = (height_ - 2) % num_threads;

  size_t start_row = 1;

  for (unsigned int i = 0; i < num_threads; ++i) {
    size_t extra_row = (i < remainder) ? 1 : 0;
    size_t end_row = start_row + rows_per_thread + extra_row;

    threads.emplace_back(&TaskSTL::ProcessImageStripe, this, start_row, end_row);

    start_row = end_row;
  }

  for (auto &thread : threads) {
    thread.join();
  }

  return true;
}

bool petrov_o_vertical_image_filtration_stl::TaskSTL::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
