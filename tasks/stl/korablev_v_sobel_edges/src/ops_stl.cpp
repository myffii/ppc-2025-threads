#include "stl/korablev_v_sobel_edges/include/ops_stl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

// clang-format off
constexpr int8_t kSobelKernelX[3][3] = {
  {-1, 0, 1},
  {-2, 0, 2},
  {-1, 0, 1}
};
constexpr int8_t kSobelKernelY[3][3] = {
  {-1, -2, -1},
  { 0,  0,  0},
  { 1,  2,  1}
};
// clang-format on

void korablev_v_sobel_edges_stl::Image::SetDimensions(std::size_t w,  // NOLINT(bugprone-easily-swappable-parameters)
                                                      std::size_t h) {
  width = w;
  height = h;
  data.resize(width * height * kPixelChannels);
}
void korablev_v_sobel_edges_stl::Image::CopyFrom(uint8_t* buf) {
  data.assign(buf, buf + (width * height * kPixelChannels));
}

bool korablev_v_sobel_edges_stl::TestTask::ValidationImpl() {
  const auto width = task_data->inputs_count[0];
  const auto height = task_data->inputs_count[1];
  return width > 0 && height > 0 && task_data->outputs_count[0] == (width * height * Image::kPixelChannels);
}

bool korablev_v_sobel_edges_stl::TestTask::PreProcessingImpl() {
  in_.SetDimensions(task_data->inputs_count[0], task_data->inputs_count[1]);
  in_.CopyFrom(task_data->inputs[0]);
  out_.SetDimensions(task_data->inputs_count[0], task_data->inputs_count[1]);
  return true;
}

bool korablev_v_sobel_edges_stl::TestTask::RunImpl() {  // NOLINT(readability-function-cognitive-complexity)
  const auto width = in_.width;
  const auto height = in_.height;

  auto& image = in_.data;

  std::vector<std::thread> threads(ppc::util::GetPPCNumThreads());
  const std::size_t delta = (height - 2) / threads.size();
  const std::size_t extra = (height - 2) % threads.size();

  std::size_t pos = 1;
  for (std::size_t threadnum = 0; threadnum < threads.size(); threadnum++) {
    const std::size_t dedicated = delta + ((threadnum < extra) ? 1 : 0);
    threads[threadnum] = std::thread(
        [&](uint32_t lidx, uint32_t ridx) {
          for (std::size_t y = lidx; y < ridx; ++y) {
            for (std::size_t x = 1; x < width - 1; ++x) {
              std::array<int32_t, 3> sum_x{0};
              std::array<int32_t, 3> sum_y{0};

              for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                  int idx = ((y + ky) * width + (x + kx)) * 3;  // NOLINT(bugprone-narrowing-conversions)
                  for (int j = 0; j < 3; j++) {
                    const int32_t pixel_value = image[idx + j];
                    sum_x[j] += kSobelKernelX[ky + 1][kx + 1] * pixel_value;
                    sum_y[j] += kSobelKernelY[ky + 1][kx + 1] * pixel_value;
                  }
                }
              }

              for (int i = 0; i < 3; ++i) {
                out_.data[((y * width + x) * 3) + i] = static_cast<uint8_t>(
                    std::min(static_cast<int32_t>(std::sqrt((sum_x[i] * sum_x[i]) + (sum_y[i] * sum_y[i]))), 255));
              }
            }
          }
        },
        pos, pos + dedicated);
    pos += dedicated;
  }
  std::ranges::for_each(threads, [](auto& thread) { thread.join(); });

  return true;
}

bool korablev_v_sobel_edges_stl::TestTask::PostProcessingImpl() {
  std::ranges::copy(out_.data, task_data->outputs[0]);
  return true;
}
