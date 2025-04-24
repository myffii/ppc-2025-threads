#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/fyodorov_m_shell_sort_with_even_odd_batcher_merge/include/ops_omp.hpp"

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp, test_small_array) {
  std::vector<int> input = {35, 33, 42, 10, 14, 19, 27, 44};
  std::vector<int> expected_output = {10, 14, 19, 27, 33, 35, 42, 44};

  std::vector<int> output(input.size(), 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp::TestTaskOpenmp test_task_openmp(task_data_omp);
  ASSERT_EQ(test_task_openmp.Validation(), true);
  test_task_openmp.PreProcessing();
  test_task_openmp.Run();
  test_task_openmp.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp, test_small_array_with_negative_values) {
  std::vector<int> input = {35, 33, 42, -10, -14, 19, 27, 44};
  std::vector<int> expected_output = {-14, -10, 19, 27, 33, 35, 42, 44};

  std::vector<int> output(input.size(), 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp::TestTaskOpenmp test_task_openmp(task_data_omp);
  ASSERT_EQ(test_task_openmp.Validation(), true);
  test_task_openmp.PreProcessing();
  test_task_openmp.Run();
  test_task_openmp.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp, test_random_sequence) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-1000, 1000);

  const size_t size = 100;
  std::vector<int> input(size);
  for (auto &num : input) {
    num = distrib(gen);
  }

  std::vector<int> expected_output = input;
  std::ranges::sort(expected_output);

  std::vector<int> output(input.size(), 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp::TestTaskOpenmp test_task_openmp(task_data_omp);
  ASSERT_EQ(test_task_openmp.Validation(), true);
  test_task_openmp.PreProcessing();
  test_task_openmp.Run();
  test_task_openmp.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp, test_sorted_array) {
  std::vector<int> input = {10, 14, 19, 27, 33, 35, 42, 44};

  std::vector<int> expected_output = {10, 14, 19, 27, 33, 35, 42, 44};

  std::vector<int> output(input.size(), 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp::TestTaskOpenmp test_task_openmp(task_data_omp);
  ASSERT_EQ(test_task_openmp.Validation(), true);
  test_task_openmp.PreProcessing();
  test_task_openmp.Run();
  test_task_openmp.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp, test_empty_array) {
  std::vector<int> input;
  std::vector<int> expected_output;
  std::vector<int> output;

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_omp->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp::TestTaskOpenmp test_task_openmp(task_data_omp);
  ASSERT_EQ(test_task_openmp.Validation(), true);
  test_task_openmp.PreProcessing();
  test_task_openmp.Run();
  test_task_openmp.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp, reverse_pozitive_array) {
  int arrsize = 12;
  std::vector<int> in = {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> out(arrsize, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp::TestTaskOpenmp test_task_openmp(task_data_omp);
  ASSERT_EQ(test_task_openmp.Validation(), true);
  test_task_openmp.PreProcessing();
  test_task_openmp.Run();
  test_task_openmp.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp, reverse_negative_array) {
  int arrsize = 12;
  std::vector<int> in = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12};
  std::vector<int> out(arrsize, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp::TestTaskOpenmp test_task_openmp(task_data_omp);
  ASSERT_EQ(test_task_openmp.Validation(), true);
  test_task_openmp.PreProcessing();
  test_task_openmp.Run();
  test_task_openmp.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp, reverse_pozitive_double) {
  int arrsize = 10;
  std::vector<int> in = {5, 4, 3, 2, 1, 5, 4, 3, 2, 1};
  std::vector<int> out(arrsize, 0);
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp::TestTaskOpenmp test_task_openmp(task_data_omp);
  ASSERT_EQ(test_task_openmp.Validation(), true);
  test_task_openmp.PreProcessing();
  test_task_openmp.Run();
  test_task_openmp.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp, single_element) {
  int arrsize = 1;
  std::vector<int> in = {1};
  std::vector<int> out(arrsize, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp::TestTaskOpenmp test_task_openmp(task_data_omp);
  ASSERT_EQ(test_task_openmp.Validation(), true);
  test_task_openmp.PreProcessing();
  test_task_openmp.Run();
  test_task_openmp.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp, zero_array) {
  int arrsize = 7;
  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0};
  std::vector<int> out(arrsize, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp::TestTaskOpenmp test_task_openmp(task_data_omp);
  ASSERT_EQ(test_task_openmp.Validation(), true);
  test_task_openmp.PreProcessing();
  test_task_openmp.Run();
  test_task_openmp.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}

TEST(BatcherMergeTest, HandlesDuplicateElements) {
  std::vector<int> left = {1, 2, 2};
  std::vector<int> right = {2, 3, 4};
  std::vector<int> result(6);
  fyodorov_m_shell_sort_with_even_odd_batcher_merge_omp::TestTaskOpenmp::CallBatcherMerge(left, right, result);
  EXPECT_EQ(result, std::vector<int>({1, 2, 2, 2, 3, 4}));
}
