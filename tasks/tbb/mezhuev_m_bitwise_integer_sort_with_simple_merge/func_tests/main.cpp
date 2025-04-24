#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/mezhuev_m_bitwise_integer_sort_with_simple_merge/include/ops_tbb.hpp"

TEST(mezhuev_m_bitwise_integer_sort_tbb, test_sort_ascending) {
  constexpr size_t kCount = 100;

  std::vector<int> in(kCount, 0);
  std::vector<int> out(kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = static_cast<int>(i);
  }

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  mezhuev_m_bitwise_integer_sort_tbb::SortTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);

  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_EQ(in, out);
}

TEST(mezhuev_m_bitwise_integer_sort_tbb, test_sort_random) {
  constexpr size_t kCount = 100;

  std::vector<int> in(kCount, 0);
  std::vector<int> out(kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = rand() % 10000;
  }

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  mezhuev_m_bitwise_integer_sort_tbb::SortTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);

  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(mezhuev_m_bitwise_integer_sort_tbb, test_sort_empty) {
  std::vector<int> in(0, 0);
  std::vector<int> out(0, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  mezhuev_m_bitwise_integer_sort_tbb::SortTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);

  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_EQ(in, out);
}

TEST(mezhuev_m_bitwise_integer_sort_tbb, test_sort_reverse) {
  constexpr size_t kCount = 100;

  std::vector<int> in(kCount, 0);
  std::vector<int> out(kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = static_cast<int>(kCount - i - 1);
  }

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  mezhuev_m_bitwise_integer_sort_tbb::SortTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);

  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(mezhuev_m_bitwise_integer_sort_tbb, test_sort_large) {
  constexpr size_t kCount = 10000;

  std::vector<int> in(kCount, 0);
  std::vector<int> out(kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = rand() % 100000;
  }

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  mezhuev_m_bitwise_integer_sort_tbb::SortTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);

  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(mezhuev_m_bitwise_integer_sort_tbb, test_sort_all_equal) {
  constexpr size_t kCount = 100;

  std::vector<int> in(kCount, 5);
  std::vector<int> out(kCount, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  mezhuev_m_bitwise_integer_sort_tbb::SortTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);

  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_EQ(in, out);
}

TEST(mezhuev_m_bitwise_integer_sort_tbb, test_sort_all_negative) {
  constexpr size_t kCount = 100;

  std::vector<int> in(kCount, -5);
  std::vector<int> out(kCount, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  mezhuev_m_bitwise_integer_sort_tbb::SortTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);

  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_EQ(in, out);
}

TEST(mezhuev_m_bitwise_integer_sort_tbb, test_sort_zeros) {
  constexpr size_t kCount = 100;

  std::vector<int> in(kCount, 0);
  std::vector<int> out(kCount, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  mezhuev_m_bitwise_integer_sort_tbb::SortTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);

  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_EQ(in, out);
}

TEST(mezhuev_m_bitwise_integer_sort_tbb, test_sort_single_element) {
  constexpr size_t kCount = 1;

  std::vector<int> in(kCount, 100);
  std::vector<int> out(kCount, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  mezhuev_m_bitwise_integer_sort_tbb::SortTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);

  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  EXPECT_EQ(in, out);
}

TEST(mezhuev_m_bitwise_integer_sort_tbb, test_sort_mixed_numbers) {
  constexpr size_t kCount = 100;

  std::vector<int> in(kCount, 0);
  std::vector<int> out(kCount, 0);

  for (size_t i = 0; i < kCount / 2; i++) {
    in[i] = rand() % 10000;
    in[i + (kCount / 2)] = -(rand() % 10000);
  }

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  mezhuev_m_bitwise_integer_sort_tbb::SortTBB test_task_tbb(task_data_tbb);

  ASSERT_EQ(test_task_tbb.Validation(), true);

  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}
