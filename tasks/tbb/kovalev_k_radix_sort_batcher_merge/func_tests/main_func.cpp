#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/kovalev_k_radix_sort_batcher_merge/include/header.hpp"

const long long int kMinLl = std::numeric_limits<long long>::lowest(), kMaxLl = std::numeric_limits<long long>::max();

TEST(kovalev_k_radix_sort_batcher_merge_tbb, zero_length) {
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_FALSE(test_task_tbb.ValidationImpl());
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, not_equal_lengths) {
  const unsigned int length = 10;
  std::vector<long long int> in(length);
  std::vector<long long int> out(2 * length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_FALSE(test_task_tbb.ValidationImpl());
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, Test_No_viol_10_int) {
  const unsigned int length = 10;
  std::srand(std::time(nullptr));
  const long long int alpha = rand();
  std::vector<long long int> in(length, alpha);
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, Test_2_int) {
  const unsigned int length = 2;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, Test_5_int) {
  const unsigned int length = 5;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, Test_13_int) {
  const unsigned int length = 13;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, Test_139_int) {
  const unsigned int length = 139;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, Test_449_int) {
  const unsigned int length = 449;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, Test_793_int) {
  const unsigned int length = 793;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, Test_1000_int) {
  const unsigned int length = 1000;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, Test_2158_int) {
  const unsigned int length = 2158;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, Test_4763_int) {
  const unsigned int length = 4763;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, Test_5000_int) {
  const unsigned int length = 5000;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, Test_178892_int) {
  const unsigned int length = 178892;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, Test_215718_int) {
  const unsigned int length = 215718;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, Test_2000000_int) {
  const unsigned int length = 2000000;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, Test_244852_int) {
  const unsigned int length = 244852;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_tbb, Test_875014_int) {
  const unsigned int length = 875014;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}