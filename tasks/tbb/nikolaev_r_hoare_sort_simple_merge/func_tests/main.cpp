#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/nikolaev_r_hoare_sort_simple_merge/include/ops_tbb.hpp"

namespace {
std::vector<double> GenerateRandomVector(size_t len, double min_val = -1000.0, double max_val = 1000.0) {
  std::vector<double> vect(len);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(min_val, max_val);

  for (size_t i = 0; i < len; ++i) {
    vect[i] = dis(gen);
  }

  return vect;
}

class HoareSortTest : public testing::TestWithParam<size_t> {
 protected:
  static void CreateTest(size_t len) {
    std::vector<double> in = GenerateRandomVector(len);
    std::vector<double> out(len, 0.0);

    auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
    task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_tbb->inputs_count.emplace_back(in.size());
    task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_tbb->outputs_count.emplace_back(out.size());

    nikolaev_r_hoare_sort_simple_merge_tbb::HoareSortSimpleMergeTBB hoare_sort_simple_merge_tbb(task_data_tbb);
    ASSERT_TRUE(hoare_sort_simple_merge_tbb.Validation());
    ASSERT_TRUE(hoare_sort_simple_merge_tbb.PreProcessing());
    ASSERT_TRUE(hoare_sort_simple_merge_tbb.Run());
    ASSERT_TRUE(hoare_sort_simple_merge_tbb.PostProcessing());

    std::vector<double> ref(len);
    std::ranges::copy(in, ref.begin());
    std::ranges::sort(ref);

    EXPECT_EQ(out, ref);
  }
};

TEST_P(HoareSortTest, sort_test) { CreateTest(GetParam()); }

INSTANTIATE_TEST_SUITE_P(nikolaev_r_hoare_sort_simple_merge_seq, HoareSortTest,
                         testing::Values(1, 2, 10, 100, 150, 200, 1000, 2000, 5000));

}  // namespace

TEST(nikolaev_r_hoare_sort_simple_merge_tbb, test_empty_vect) {
  std::vector<double> in = {};
  std::vector<double> out = {};

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs_count.emplace_back(out.size());

  nikolaev_r_hoare_sort_simple_merge_tbb::HoareSortSimpleMergeTBB hoare_sort_simple_merge_tbb(task_data_tbb);
  ASSERT_FALSE(hoare_sort_simple_merge_tbb.Validation());
}

TEST(nikolaev_r_hoare_sort_simple_merge_tbb, test_reverse_order) {
  std::vector<double> in = {10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
  std::vector<double> out(in.size(), 0.0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  nikolaev_r_hoare_sort_simple_merge_tbb::HoareSortSimpleMergeTBB hoare_sort_simple_merge_tbb(task_data_tbb);
  ASSERT_TRUE(hoare_sort_simple_merge_tbb.Validation());
  ASSERT_TRUE(hoare_sort_simple_merge_tbb.PreProcessing());
  ASSERT_TRUE(hoare_sort_simple_merge_tbb.Run());
  ASSERT_TRUE(hoare_sort_simple_merge_tbb.PostProcessing());

  std::vector<double> ref(in.size());
  std::ranges::copy(in, ref.begin());
  std::ranges::sort(ref);

  EXPECT_EQ(out, ref);
}

TEST(nikolaev_r_hoare_sort_simple_merge_tbb, test_invalid_output) {
  std::vector<double> in = {1.0, 2.0, 3.0};
  std::vector<double> out = {};

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  nikolaev_r_hoare_sort_simple_merge_tbb::HoareSortSimpleMergeTBB hoare_sort_simple_merge_tbb(task_data_tbb);
  ASSERT_FALSE(hoare_sort_simple_merge_tbb.Validation());
}

TEST(nikolaev_r_hoare_sort_simple_merge_tbb, test_input_and_output_sizes_not_equal) {
  std::vector<double> in = {1.0, 2.0, 3.0};
  std::vector<double> out = {0.0, 0.0};

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  nikolaev_r_hoare_sort_simple_merge_tbb::HoareSortSimpleMergeTBB hoare_sort_simple_merge_tbb(task_data_tbb);
  ASSERT_FALSE(hoare_sort_simple_merge_tbb.Validation());
}
