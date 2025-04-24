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
#include "omp/kovalev_k_radix_sort_batcher_merge/include/header.hpp"

const long long int kMinLl = std::numeric_limits<long long>::lowest(), kMaxLl = std::numeric_limits<long long>::max();

TEST(kovalev_k_radix_sort_batcher_merge_omp, zero_length) {
  std::vector<long long int> in;
  std::vector<long long int> out;
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_FALSE(test_task_omp.ValidationImpl());
}

TEST(kovalev_k_radix_sort_batcher_merge_omp, not_equal_lengths) {
  const unsigned int length = 10;
  std::vector<long long int> in(length);
  std::vector<long long int> out(2 * length);
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_FALSE(test_task_omp.ValidationImpl());
}

TEST(kovalev_k_radix_sort_batcher_merge_omp, Test_No_viol_10_int) {
  const unsigned int length = 10;
  std::srand(std::time(nullptr));
  const long long int alpha = rand();
  std::vector<long long int> in(length, alpha);
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_omp, Test_2_int) {
  const unsigned int length = 2;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_omp, Test_5_int) {
  const unsigned int length = 5;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_omp, Test_793_int) {
  const unsigned int length = 793;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_omp, Test_1000_int) {
  const unsigned int length = 1000;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_omp, Test_2158_int) {
  const unsigned int length = 2158;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_omp, Test_4763_int) {
  const unsigned int length = 4763;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_omp, Test_5000_int) {
  const unsigned int length = 5000;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_omp, Test_178892_int) {
  const unsigned int length = 178892;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_omp, Test_215718_int) {
  const unsigned int length = 215718;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_omp, Test_2000000_int) {
  const unsigned int length = 2000000;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_omp, Test_244852_int) {
  const unsigned int length = 244852;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_omp, Test_875014_int) {
  const unsigned int length = 875014;
  std::vector<long long int> in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  std::ranges::sort(in.begin(), in.end(), [](long long int a, long long int b) { return a < b; });
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (out[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}