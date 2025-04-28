#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/belov_a_radix_sort_with_batcher_mergesort/include/ops_tbb.hpp"

using namespace belov_a_radix_batcher_mergesort_tbb;

namespace {
std::vector<Bigint> GenerateMixedValuesArray(int n) {
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<Bigint> small_range(-999LL, 999LL);
  std::uniform_int_distribution<Bigint> medium_range(-10000LL, 10000LL);
  std::uniform_int_distribution<Bigint> large_range(-10000000000LL, 10000000000LL);
  std::uniform_int_distribution<int> choice(0, 2);

  std::vector<Bigint> arr;
  arr.reserve(n);

  for (int i = 0; i < n; ++i) {
    int c = choice(gen);
    if (c == 0) {
      arr.push_back(large_range(gen));
    } else if (c == 1) {
      arr.push_back(medium_range(gen));
    } else {
      arr.push_back(small_range(gen));
    }
  }
  return arr;
}

std::vector<Bigint> GenerateIntArray(int n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());

  std::vector<Bigint> arr;
  arr.reserve(n);

  for (int i = 0; i < n; ++i) {
    arr.push_back(dist(gen));
  }
  return arr;
}

std::vector<Bigint> GenerateBigintArray(int n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<Bigint> dist(std::numeric_limits<Bigint>::min() / 2,
                                             std::numeric_limits<Bigint>::max() / 2);

  std::vector<Bigint> arr;
  arr.reserve(n);

  for (int i = 0; i < n; ++i) {
    arr.push_back(dist(gen));
  }
  return arr;
}
}  // namespace

TEST(belov_a_radix_batcher_mergesort_tbb, test_random_small_BigintV_vector) {
  int n = 1024;
  std::vector<Bigint> arr = GenerateBigintArray(n);

  std::vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->inputs_count.emplace_back(arr.size());
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_tbb(task_data_tbb);
  ASSERT_EQ(tesk_task_tbb.Validation(), true);
  tesk_task_tbb.PreProcessing();
  tesk_task_tbb.Run();
  tesk_task_tbb.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_tbb, test_random_medium_BigintV_vector) {
  int n = 8192;
  std::vector<Bigint> arr = GenerateBigintArray(n);

  std::vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->inputs_count.emplace_back(arr.size());
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_tbb(task_data_tbb);
  ASSERT_EQ(tesk_task_tbb.Validation(), true);
  tesk_task_tbb.PreProcessing();
  tesk_task_tbb.Run();
  tesk_task_tbb.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_tbb, test_random_large_BigintV_vector) {
  int n = 65536;
  std::vector<Bigint> arr = GenerateBigintArray(n);

  std::vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->inputs_count.emplace_back(arr.size());
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_tbb(task_data_tbb);
  ASSERT_EQ(tesk_task_tbb.Validation(), true);
  tesk_task_tbb.PreProcessing();
  tesk_task_tbb.Run();
  tesk_task_tbb.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_tbb, test_random_small_intV_vector) {
  int n = 4096;
  std::vector<Bigint> arr = GenerateIntArray(n);

  std::vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->inputs_count.emplace_back(arr.size());
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_tbb(task_data_tbb);
  ASSERT_EQ(tesk_task_tbb.Validation(), true);
  tesk_task_tbb.PreProcessing();
  tesk_task_tbb.Run();
  tesk_task_tbb.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_tbb, test_random_medium_intV_vector) {
  int n = 16384;
  std::vector<Bigint> arr = GenerateIntArray(n);

  std::vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->inputs_count.emplace_back(arr.size());
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_tbb(task_data_tbb);
  ASSERT_EQ(tesk_task_tbb.Validation(), true);
  tesk_task_tbb.PreProcessing();
  tesk_task_tbb.Run();
  tesk_task_tbb.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_tbb, test_random_large_intV_vector) {
  int n = 65536;
  std::vector<Bigint> arr = GenerateIntArray(n);

  std::vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->inputs_count.emplace_back(arr.size());
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_tbb(task_data_tbb);
  ASSERT_EQ(tesk_task_tbb.Validation(), true);
  tesk_task_tbb.PreProcessing();
  tesk_task_tbb.Run();
  tesk_task_tbb.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_tbb, test_random_small_mixedV_vector) {
  int n = 2048;
  std::vector<Bigint> arr = GenerateMixedValuesArray(n);

  std::vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->inputs_count.emplace_back(arr.size());
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_tbb(task_data_tbb);
  ASSERT_EQ(tesk_task_tbb.Validation(), true);
  tesk_task_tbb.PreProcessing();
  tesk_task_tbb.Run();
  tesk_task_tbb.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_tbb, test_random_medium_mixedV_vector) {
  int n = 16384;
  std::vector<Bigint> arr = GenerateMixedValuesArray(n);

  std::vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->inputs_count.emplace_back(arr.size());
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_tbb(task_data_tbb);
  ASSERT_EQ(tesk_task_tbb.Validation(), true);
  tesk_task_tbb.PreProcessing();
  tesk_task_tbb.Run();
  tesk_task_tbb.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_tbb, test_random_large_mixedV_vector) {
  int n = 65536;
  std::vector<Bigint> arr = GenerateMixedValuesArray(n);

  std::vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->inputs_count.emplace_back(arr.size());
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_tbb(task_data_tbb);
  ASSERT_EQ(tesk_task_tbb.Validation(), true);
  tesk_task_tbb.PreProcessing();
  tesk_task_tbb.Run();
  tesk_task_tbb.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_tbb, test_predefined_intV_vector) {
  int n = 16;
  std::vector<Bigint> arr = {74685421,  -53749, 2147483647, -1000, -2147483648, 1001, 0,       124,
                             315986930, -123,   42,         -43,   2,           -1,   -999999, 999998};

  std::vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->inputs_count.emplace_back(arr.size());
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_tbb(task_data_tbb);
  ASSERT_EQ(tesk_task_tbb.Validation(), true);
  tesk_task_tbb.PreProcessing();
  tesk_task_tbb.Run();
  tesk_task_tbb.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_tbb, test_one_element_input_Bigint) {
  int n = 1;
  std::vector<Bigint> arr = {8888};

  std::vector<Bigint> expected_solution = arr;

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->inputs_count.emplace_back(arr.size());
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_tbb(task_data_tbb);
  ASSERT_EQ(tesk_task_tbb.Validation(), true);
  tesk_task_tbb.PreProcessing();
  tesk_task_tbb.Run();
  tesk_task_tbb.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_tbb, test_array_size_missmatch) {
  int n = 3;
  std::vector<Bigint> arr = {-53742329, -2147483648, 123265244, 0, 315986930, 42, 2147483647, -853960, 472691};

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->inputs_count.emplace_back(arr.size());
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_tbb(task_data_tbb);

  EXPECT_FALSE(tesk_task_tbb.Validation());
}

TEST(belov_a_radix_batcher_mergesort_tbb, test_invalid_inputs_count) {
  int n = 3;
  std::vector<Bigint> arr = {-53742329, -2147483648, 123265244, 0, 315986930, 42, 2147483647, -853960, 472691};

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_tbb(task_data_tbb);

  EXPECT_FALSE(tesk_task_tbb.Validation());
}

TEST(belov_a_radix_batcher_mergesort_tbb, test_empty_input_Validation) {
  int n = 0;
  std::vector<Bigint> arr = {};

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->inputs_count.emplace_back(arr.size());
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_tbb(task_data_tbb);

  EXPECT_FALSE(tesk_task_tbb.Validation());
}

TEST(belov_a_radix_batcher_mergesort_tbb, test_empty_output_Validation) {
  int n = 3;
  std::vector<Bigint> arr = {789, 654, 231, 0, 123456789, 792012345678, -22475942, -853960, 59227648};

  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_tbb->inputs_count.emplace_back(arr.size());
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_tbb(task_data_tbb);

  EXPECT_FALSE(tesk_task_tbb.Validation());
}