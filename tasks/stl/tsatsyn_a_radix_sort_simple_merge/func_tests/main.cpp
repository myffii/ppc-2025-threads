#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/tsatsyn_a_radix_sort_simple_merge/include/ops_stl.hpp"

namespace {
std::vector<double> GetRandomVector(int sz, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(a, b);
  std::vector<double> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dis(gen);
  }
  return vec;
}
}  // namespace
TEST(tsatsyn_a_radix_sort_simple_merge_stl, negative_double_10) {
  // Create data
  int arrsize = 10;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_stl, negative_double_100) {
  // Create data
  int arrsize = 100;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_stl, negative_double_1000) {
  // Create data
  int arrsize = 1000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_stl, negative_double_10000) {
  // Create data
  int arrsize = 10000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_stl, negative_double_100000) {
  // Create data
  int arrsize = 100000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}

TEST(tsatsyn_a_radix_sort_simple_merge_stl, pozitive_double_10) {
  // Create data
  int arrsize = 10;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_stl, pozitive_double_100) {
  // Create data
  int arrsize = 100;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_stl, pozitive_double_1000) {
  // Create data
  int arrsize = 1000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_stl, pozitive_double_10000) {
  // Create data
  int arrsize = 10000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_stl, pozitive_double_100000) {
  // Create data
  int arrsize = 100000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_stl, reverse_pozitive_double_10) {
  // Create data
  int arrsize = 10;
  std::vector<double> in = {5.0, 4.0, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  std::vector<double> out(arrsize, 0);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
