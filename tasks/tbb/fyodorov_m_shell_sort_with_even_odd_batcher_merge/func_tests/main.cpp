#include <gtest/gtest.h>
#include <oneapi/tbb/global_control.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/fyodorov_m_shell_sort_with_even_odd_batcher_merge/include/ops_tbb.hpp"

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb, test_small_array) {
  tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, 4);
  std::vector<int> input = {35, 33, 42, 10, 14, 19, 27, 44};
  std::vector<int> expected_output = {10, 14, 19, 27, 33, 35, 42, 44};

  std::vector<int> output(input.size(), 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_tbb->inputs_count.emplace_back(input.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_tbb->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb, test_small_array_with_negative_values) {
  tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, 4);
  std::vector<int> input = {35, 33, 42, -10, -14, 19, 27, 44};
  std::vector<int> expected_output = {-14, -10, 19, 27, 33, 35, 42, 44};

  std::vector<int> output(input.size(), 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_tbb->inputs_count.emplace_back(input.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_tbb->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb, test_random_sequence) {
  tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, 4);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-100, 100);

  const size_t size = 10;
  std::vector<int> input(size);
  for (auto &num : input) {
    num = distrib(gen);
  }

  std::vector<int> expected_output = input;
  std::ranges::sort(expected_output);

  std::vector<int> output(input.size(), 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_tbb->inputs_count.emplace_back(input.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_tbb->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb, test_sorted_array) {
  tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, 4);
  std::vector<int> input = {10, 14, 19, 27, 33, 35, 42, 44};

  std::vector<int> expected_output = {10, 14, 19, 27, 33, 35, 42, 44};

  std::vector<int> output(input.size(), 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_tbb->inputs_count.emplace_back(input.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_tbb->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb, test_empty_array) {
  tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, 4);
  std::vector<int> input;
  std::vector<int> expected_output;
  std::vector<int> output;

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_tbb->inputs_count.emplace_back(input.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_tbb->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb, reverse_pozitive_array) {
  tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, 4);
  int arrsize = 12;
  std::vector<int> in = {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> out(arrsize, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb, reverse_pozitive_double) {
  tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, 4);
  int arrsize = 10;
  std::vector<int> in = {5, 4, 3, 2, 1, 5, 4, 3, 2, 1};
  std::vector<int> out(arrsize, 0);
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb, test_validation_failure) {
  tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, 4);

  std::vector<int> input = {35, 33, 42, 10};
  std::vector<int> output;

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_tbb->inputs_count.emplace_back(input.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_tbb->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), false);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb, test_empty_array_validation) {
  tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, 4);

  std::vector<int> input;
  std::vector<int> output;

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_tbb->inputs_count.emplace_back(input.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_tbb->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);

  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_EQ(output, input);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb, test_single_element_array) {
  tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, 4);

  std::vector<int> input = {42};
  std::vector<int> expected_output = {42};
  std::vector<int> output(input.size(), 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_tbb->inputs_count.emplace_back(input.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_tbb->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);

  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb, HandlesDuplicateElements) {
  std::vector<int> left = {1, 2, 2};
  std::vector<int> right = {2, 3, 4};
  std::vector<int> result(6);
  fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb::TestTaskTBB::CallBatcherMerge(left, right, result);
  EXPECT_EQ(result, std::vector<int>({1, 2, 2, 2, 3, 4}));
}