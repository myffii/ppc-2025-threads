#include "tbb/zaytsev_d_sobel/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"

bool zaytsev_d_sobel_tbb::TestTaskTBB::PreProcessingImpl() {
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + task_data->inputs_count[0]);

  auto *size_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  width_ = size_ptr[0];
  height_ = size_ptr[1];

  output_ = std::vector<int>(task_data->outputs_count[0], 0);
  return true;
}

bool zaytsev_d_sobel_tbb::TestTaskTBB::ValidationImpl() {
  auto *size_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  int width = size_ptr[0];
  int height = size_ptr[1];
  return (task_data->inputs_count[0] == task_data->outputs_count[0]) && (width >= 3) && (height >= 3) &&
         ((width * height) == int(task_data->inputs_count[0]));
}

bool zaytsev_d_sobel_tbb::TestTaskTBB::RunImpl() {
  const int gxkernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int gykernel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  int rows = height_ - 2;
  int cols = width_ - 2;
  int total_pixels = rows * cols;

  tbb::parallel_for(tbb::blocked_range<int>(0, total_pixels),
                    [this, &gxkernel, &gykernel, cols](const tbb::blocked_range<int> &r) {
                      for (int index = r.begin(); index < r.end(); ++index) {
                        int i = 1 + (index / cols);
                        int j = 1 + (index % cols);

                        int sumgx = 0;
                        int sumgy = 0;

                        for (int di = -1; di <= 1; ++di) {
                          for (int dj = -1; dj <= 1; ++dj) {
                            int ni = i + di;
                            int nj = j + dj;
                            int kernel_row = di + 1;
                            int kernel_col = dj + 1;

                            sumgx += input_[(ni * width_) + nj] * gxkernel[kernel_row][kernel_col];
                            sumgy += input_[(ni * width_) + nj] * gykernel[kernel_row][kernel_col];
                          }
                        }

                        int magnitude = static_cast<int>(std::sqrt((sumgx * sumgx) + (sumgy * sumgy)));
                        output_[(i * width_) + j] = std::min(magnitude, 255);
                      }
                    });

  return true;
}

bool zaytsev_d_sobel_tbb::TestTaskTBB::PostProcessingImpl() {
  std::ranges::copy(output_, reinterpret_cast<int *>(task_data->outputs[0]));
  return true;
}
