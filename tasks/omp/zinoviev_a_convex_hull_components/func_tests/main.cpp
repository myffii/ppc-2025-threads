#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/zinoviev_a_convex_hull_components/include/ops_omp.hpp"

namespace {
void SetupTest(std::shared_ptr<ppc::core::TaskData>& data, const std::vector<int>& input, int w, int h,
               size_t out_size) {
  data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(input.data())));
  data->inputs_count.push_back(w);
  data->inputs_count.push_back(h);
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_omp::Point[out_size]));
  data->outputs_count.push_back(static_cast<int>(out_size));
}

void CheckResult(const std::vector<zinoviev_a_convex_hull_components_omp::Point>& result,
                 const std::vector<zinoviev_a_convex_hull_components_omp::Point>& expect) {
  ASSERT_EQ(result.size(), expect.size());
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i].x, expect[i].x);
    ASSERT_EQ(result[i].y, expect[i].y);
  }
}

void RunTest(const std::vector<int>& input, const std::vector<zinoviev_a_convex_hull_components_omp::Point>& expect,
             int w, int h) {
  std::shared_ptr<ppc::core::TaskData> data;
  SetupTest(data, input, w, h, expect.size());

  zinoviev_a_convex_hull_components_omp::ConvexHullOMP task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto* res = reinterpret_cast<zinoviev_a_convex_hull_components_omp::Point*>(data->outputs[0]);
  std::vector<zinoviev_a_convex_hull_components_omp::Point> actual(res, res + expect.size());
  CheckResult(actual, expect);
  delete[] res;
}
}  // namespace

TEST(zinoviev_a_convex_hull_omp, EmptyImage) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  std::vector<int> input(25, 0);
  RunTest(input, {}, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_omp, FullRectangle) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  std::vector<int> input(25, 1);
  const std::vector<zinoviev_a_convex_hull_components_omp::Point> expect{
      {.x = 0, .y = 0}, {.x = 4, .y = 0}, {.x = 3, .y = 4}, {.x = 0, .y = 4}};
  RunTest(input, expect, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_omp, CrossShape) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  std::vector<int> input = {0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0};
  const std::vector<zinoviev_a_convex_hull_components_omp::Point> expect{
      {.x = 0, .y = 1}, {.x = 1, .y = 0}, {.x = 4, .y = 1}, {.x = 1, .y = 4}};
  RunTest(input, expect, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_omp, SinglePoint) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  std::vector<int> input(25, 0);
  input[12] = 1;
  const std::vector<zinoviev_a_convex_hull_components_omp::Point> expect{{.x = 2, .y = 2}};
  RunTest(input, expect, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_omp, FullSquare) {
  constexpr int kWidth = 3;
  constexpr int kHeight = 3;
  std::vector<int> input(9, 1);
  const std::vector<zinoviev_a_convex_hull_components_omp::Point> expect{
      {.x = 0, .y = 0}, {.x = 2, .y = 0}, {.x = 1, .y = 2}, {.x = 0, .y = 2}};
  RunTest(input, expect, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_omp, LargeRectangleBorder) {
  const int size = 100;
  std::vector<int> input(size * size, 0);
  for (int i = 0; i < size; ++i) {
    input[i] = 1;
    input[((size - 1) * size) + i] = 1;
    input[i * size] = 1;
    input[(i * size) + (size - 1)] = 1;
  }
  const std::vector<zinoviev_a_convex_hull_components_omp::Point> expect{
      {.x = 0, .y = 0}, {.x = size - 1, .y = 0}, {.x = size - 2, .y = size - 1}, {.x = 0, .y = size - 1}};

  RunTest(input, expect, size, size);
}