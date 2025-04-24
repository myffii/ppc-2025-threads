#include <gtest/gtest.h>
#include <oneapi/tbb/task_arena.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "tbb/sotskov_a_shell_sorting_with_simple_merging/include/ops_tbb.hpp"

namespace sotskov_a_shell_sorting_with_simple_merging_tbb {
namespace {

struct RandomVectorParams {
  int size;
  int min_value;
  int max_value;
};
struct SortingTestParams {
  std::vector<int> expected;
  std::vector<int> input;
};

std::vector<int> GenerateRandomVector(const RandomVectorParams &params) {
  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::uniform_int_distribution<int> distribution(params.min_value, params.max_value);

  std::vector<int> random_vector(params.size);

  for (int &element : random_vector) {
    element = distribution(generator);
  }

  return random_vector;
}

void RunSortingTest(SortingTestParams &params, void (*sort_func)(std::vector<int> &)) {
  std::vector<int> out(params.input.size(), 0);

  std::vector<int> sorted_expected = params.expected;
  std::ranges::sort(sorted_expected.begin(), sorted_expected.end());

  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  arena.execute([&] { sort_func(params.input); });

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(params.input.data()));
  task_data_tbb->inputs_count.emplace_back(params.input.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  sotskov_a_shell_sorting_with_simple_merging_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.ValidationImpl(), true);
  test_task_tbb.PreProcessingImpl();
  test_task_tbb.RunImpl();
  test_task_tbb.PostProcessingImpl();

  ASSERT_EQ(out, sorted_expected);
}
}  // namespace
}  // namespace sotskov_a_shell_sorting_with_simple_merging_tbb

TEST(sotskov_a_shell_sorting_with_simple_merging_tbb, test_sort_positive_numbers) {
  sotskov_a_shell_sorting_with_simple_merging_tbb::SortingTestParams params = {.expected = {1, 1, 2, 4, 5, 6, 7, 8},
                                                                               .input = {5, 1, 8, 6, 2, 7, 1, 4}};

  sotskov_a_shell_sorting_with_simple_merging_tbb::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_tbb::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_tbb, test_sort_negative_numbers) {
  sotskov_a_shell_sorting_with_simple_merging_tbb::SortingTestParams params = {.expected = {-12, -10, -8, -7, -4, -3},
                                                                               .input = {-8, -3, -12, -7, -4, -10}};

  sotskov_a_shell_sorting_with_simple_merging_tbb::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_tbb::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_tbb, test_sort_ordered_array) {
  sotskov_a_shell_sorting_with_simple_merging_tbb::SortingTestParams params = {.expected = {1, 2, 3, 4, 5, 6, 7, 8},
                                                                               .input = {1, 2, 3, 4, 5, 6, 7, 8}};

  sotskov_a_shell_sorting_with_simple_merging_tbb::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_tbb::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_tbb, test_sort_with_duplicates) {
  sotskov_a_shell_sorting_with_simple_merging_tbb::SortingTestParams params = {.expected = {2, 2, 2, 4, 4, 6, 6, 8},
                                                                               .input = {4, 2, 2, 8, 4, 6, 6, 2}};

  sotskov_a_shell_sorting_with_simple_merging_tbb::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_tbb::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_tbb, test_sort_single_element) {
  sotskov_a_shell_sorting_with_simple_merging_tbb::SortingTestParams params = {.expected = {77}, .input = {77}};

  sotskov_a_shell_sorting_with_simple_merging_tbb::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_tbb::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_tbb, test_sort_empty_array) {
  sotskov_a_shell_sorting_with_simple_merging_tbb::SortingTestParams params = {.expected = {}, .input = {}};

  sotskov_a_shell_sorting_with_simple_merging_tbb::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_tbb::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_tbb, test_sort_random_vector) {
  sotskov_a_shell_sorting_with_simple_merging_tbb::RandomVectorParams params = {
      .size = 20, .min_value = -100, .max_value = 100};
  std::vector<int> in = sotskov_a_shell_sorting_with_simple_merging_tbb::GenerateRandomVector(params);
  std::vector<int> expected = in;

  std::ranges::sort(expected.begin(), expected.end());

  sotskov_a_shell_sorting_with_simple_merging_tbb::SortingTestParams sorting_params = {.expected = expected,
                                                                                       .input = in};

  sotskov_a_shell_sorting_with_simple_merging_tbb::RunSortingTest(
      sorting_params, sotskov_a_shell_sorting_with_simple_merging_tbb::ShellSortWithSimpleMerging);
}
TEST(sotskov_a_shell_sorting_with_simple_merging_tbb, test_sort_double_reversed_array) {
  const int size = 100;
  std::vector<int> input(size);

  for (int i = 0; i < size / 2; ++i) {
    input[i] = size / 2 - i;
  }

  for (int i = size / 2; i < size; ++i) {
    input[i] = size - i;
  }

  std::vector<int> expected = input;
  std::ranges::sort(expected.begin(), expected.end());

  sotskov_a_shell_sorting_with_simple_merging_tbb::SortingTestParams params = {.expected = expected, .input = input};

  sotskov_a_shell_sorting_with_simple_merging_tbb::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_tbb::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_tbb, test_sort_all_identical_elements) {
  sotskov_a_shell_sorting_with_simple_merging_tbb::SortingTestParams params = {
      .expected = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5}, .input = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5}};

  sotskov_a_shell_sorting_with_simple_merging_tbb::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_tbb::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_tbb, test_sort_large_array_of_identical_elements) {
  const int size = 100;
  std::vector<int> input(size, 22);
  std::vector<int> expected = input;

  sotskov_a_shell_sorting_with_simple_merging_tbb::SortingTestParams params = {.expected = expected, .input = input};

  sotskov_a_shell_sorting_with_simple_merging_tbb::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_tbb::ShellSortWithSimpleMerging);
}