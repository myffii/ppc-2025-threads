#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/gusev_n_sorting_int_simple_merging/include/ops_stl.hpp"

namespace {
ppc::core::TaskDataPtr CreateTaskData(std::vector<int>& input, std::vector<int>& output) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());
  return task_data;
}

void RunTest(std::vector<int>& input, std::vector<int>& output) {
  auto task_data = CreateTaskData(input, output);
  gusev_n_sorting_int_simple_merging_stl::TestTaskSTL task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}
}  // namespace

TEST(gusev_n_sorting_int_simple_merging_stl, test_radix_sort_basic) {
  std::vector<int> in = {17, 45, 75, 90, 8, 24, 2, 66};
  std::vector<int> out(in.size());
  RunTest(in, out);
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_stl, test_radix_sort_empty) {
  std::vector<int> in;
  std::vector<int> out;
  RunTest(in, out);
  EXPECT_TRUE(out.empty());
}

TEST(gusev_n_sorting_int_simple_merging_stl, test_radix_sort_single_element) {
  std::vector<int> in = {42};
  std::vector<int> out(1);
  RunTest(in, out);
  EXPECT_EQ(in, out);
}

TEST(gusev_n_sorting_int_simple_merging_stl, test_radix_sort_single_zero) {
  std::vector<int> in = {0};
  std::vector<int> out(1);
  RunTest(in, out);
  EXPECT_EQ(in, out);
}

TEST(gusev_n_sorting_int_simple_merging_stl, test_radix_sort_same_values) {
  std::vector<int> in = {5, 5, 5, 5, 5};
  std::vector<int> out(5);
  RunTest(in, out);
  EXPECT_EQ(in, out);
}

TEST(gusev_n_sorting_int_simple_merging_stl, test_radix_sort_all_negative) {
  std::vector<int> in = {-5, -9, -3, -5, -8};
  std::vector<int> out(5);
  RunTest(in, out);
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(gusev_n_sorting_simple_merging_stl, test_radix_sort_random) {
  std::vector<int> in(5);
  std::vector<int> out(5);
  std::ranges::generate(in, []() { return (std::rand() % 2000) - 1000; });
  RunTest(in, out);
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_stl, test_radix_sort_negative_numbers) {
  std::vector<int> in = {3, -1, 0, -5, 2, -3};
  std::vector<int> out(in.size());
  RunTest(in, out);
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_stl, test_radix_sort_duplicates) {
  std::vector<int> in = {5, 3, 5, -2, 3, -2, -2, 5, 0, 0, 7, 7, -7};
  std::vector<int> out(in.size());
  RunTest(in, out);
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_stl, test_radix_sort_reverse_order) {
  std::vector<int> in = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3};
  std::vector<int> out(in.size());
  RunTest(in, out);
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}
