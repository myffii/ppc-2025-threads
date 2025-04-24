#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/tsatsyn_a_radix_sort_simple_merge/include/ops_tbb.hpp"

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

TEST(tsatsyn_a_radix_sort_simple_merge_tbb, negative_validation) {
  // Create data
  int arrsize = 10;
  std::vector<double> in;
  std::vector<double> out(arrsize / 2, 0);
  in = GetRandomVector(arrsize, -100, 0);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), false);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, test_matmul_50) {
  constexpr int kCount = 50;

  // Create data
  std::vector<double> in;
  std::vector<double> out(kCount * kCount, 0);
  in = GetRandomVector(kCount * kCount, 0, 100);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, negative_double_10) {
  // Create data
  int arrsize = 10;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, negative_double_100) {
  // Create data
  int arrsize = 100;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, negative_double_1000) {
  // Create data
  int arrsize = 1000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, negative_double_10000) {
  // Create data
  int arrsize = 10000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, negative_double_100000) {
  // Create data
  int arrsize = 100000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}

TEST(tsatsyn_a_radix_sort_simple_merge_tbb, pozitive_double_10) {
  // Create data
  int arrsize = 10;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, pozitive_double_100) {
  // Create data
  int arrsize = 100;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, pozitive_double_1000) {
  // Create data
  int arrsize = 1000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, pozitive_double_10000) {
  // Create data
  int arrsize = 10000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, pozitive_double_100000) {
  // Create data
  int arrsize = 100000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, reverse_pozitive_double_10) {
  // Create data
  int arrsize = 10;
  std::vector<double> in = {5.0, 4.0, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  std::vector<double> out(arrsize, 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}

TEST(tsatsyn_a_radix_sort_simple_merge_tbb, equal_pozitive_double_10) {
  // Create data
  int arrsize = 10;
  std::vector<double> in = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<double> out(arrsize, 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, zero_elements_double_10) {
  // Create data
  int arrsize = 10;
  std::vector<double> in = {};
  std::vector<double> out(arrsize, 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), false);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, single_pozitive_double_10) {
  // Create data
  int arrsize = 1;
  std::vector<double> in = {1.0};
  std::vector<double> out(arrsize, 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, mix_double_10) {
  // Create data
  int arrsize = 10;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, mix_double_100) {
  // Create data
  int arrsize = 100;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);

  // Create task_data
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, mix_double_1000) {
  // Create data
  int arrsize = 1000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);

  // Create task_data
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_tbb, mix_double_10000) {
  // Create data
  int arrsize = 10000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);

  // Create task_data
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
