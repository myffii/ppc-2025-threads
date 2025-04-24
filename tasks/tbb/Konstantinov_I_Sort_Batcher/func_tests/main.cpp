#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/Konstantinov_I_Sort_Batcher/include/ops_tbb.hpp"

TEST(Konstantinov_I_Sort_Batcher_tbb, test_wrong_size) {
  std::vector<double> in(2, 0.0);
  std::vector<double> out(1);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_tbb::RadixSortBatcherTBB test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), false);
}

TEST(Konstantinov_I_Sort_Batcher_tbb, test_scalar) {
  std::vector<double> in{3.14};
  std::vector<double> exp_out{3.14};
  std::vector<double> out(1);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_tbb::RadixSortBatcherTBB test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(Konstantinov_I_Sort_Batcher_tbb, test_negative_values) {
  std::vector<double> in{-3.14, -1.0, -100.5, -0.1, -999.99};
  std::vector<double> exp_out{-999.99, -100.5, -3.14, -1.0, -0.1};
  std::vector<double> out(5);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_tbb::RadixSortBatcherTBB test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(Konstantinov_I_Sort_Batcher_tbb, test_empty_input) {
  std::vector<double> in{};
  std::vector<double> exp_out{};
  std::vector<double> out(0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_tbb::RadixSortBatcherTBB test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(Konstantinov_I_Sort_Batcher_tbb, test_duplicate_values) {
  std::vector<double> in{5.0, 3.0, 5.0, 2.0, 3.0, 5.0, 2.0};
  std::vector<double> exp_out{5.0, 3.0, 5.0, 2.0, 3.0, 5.0, 2.0};
  std::vector<double> out(7);
  std::ranges::sort(exp_out);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_tbb::RadixSortBatcherTBB test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(Konstantinov_I_Sort_Batcher_tbb, test_random_10000_values) {
  constexpr size_t kCount = 10000;
  std::vector<double> in(kCount);
  std::vector<double> out(kCount);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

  for (auto &num : in) {
    num = dist(gen);
  }
  std::vector<double> exp_out = in;
  std::ranges::sort(exp_out);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_tbb::RadixSortBatcherTBB test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(Konstantinov_I_Sort_Batcher_tbb, test_random_100000_values) {
  constexpr size_t kCount = 100000;
  std::vector<double> in(kCount);
  std::vector<double> out(kCount);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

  for (auto &num : in) {
    num = dist(gen);
  }
  std::vector<double> exp_out = in;
  std::ranges::sort(exp_out);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_tbb::RadixSortBatcherTBB test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(Konstantinov_I_Sort_Batcher_tbb, test_random_1000000_values) {
  constexpr size_t kCount = 1000000;
  std::vector<double> in(kCount);
  std::vector<double> out(kCount);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

  for (auto &num : in) {
    num = dist(gen);
  }
  std::vector<double> exp_out = in;
  std::ranges::sort(exp_out);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_tbb::RadixSortBatcherTBB test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}