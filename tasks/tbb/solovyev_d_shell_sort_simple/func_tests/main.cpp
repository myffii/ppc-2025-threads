#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/solovyev_d_shell_sort_simple/include/ops_tbb.hpp"

namespace {
std::vector<int> GetRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = (int)((gen() % (200)) - 100);
  }
  return vec;
}
bool IsSorted(std::vector<int> data) {
  int last = INT_MIN;
  for (size_t i = 0; i < data.size(); i++) {
    if (data[i] < last) {
      return false;
    }
    last = data[i];
  }
  return true;
}
}  // namespace

TEST(solovyev_d_shell_sort_simple_tbb, sort_empty) {
  // Create data
  std::vector<int> in = {};
  std::vector<int> out(in.size(), 0);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  solovyev_d_shell_sort_simple_tbb::TaskTBB task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), true);
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();
  ASSERT_TRUE(IsSorted(out));
}

TEST(solovyev_d_shell_sort_simple_tbb, sort_10_negative) {
  // Create data
  std::vector<int> in = {1, 5, -7, 3, 7, -3, 8, 4, -1, 6};
  std::vector<int> out(in.size(), 0);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  solovyev_d_shell_sort_simple_tbb::TaskTBB task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), true);
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();
  ASSERT_TRUE(IsSorted(out));
}

TEST(solovyev_d_shell_sort_simple_tbb, sort_10) {
  // Create data
  std::vector<int> in = {1, 5, 7, 3, 7, 3, 8, 4, 1, 6};
  std::vector<int> out(in.size(), 0);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  solovyev_d_shell_sort_simple_tbb::TaskTBB task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), true);
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();
  ASSERT_TRUE(IsSorted(out));
}

TEST(solovyev_d_shell_sort_simple_tbb, sort_20) {
  // Create data
  std::vector<int> in = {1, 5, 7, 3, 7, 3, 8, 4, 1, 6, 4, 6, 7, 3, 12, 21, 65, 43, 1, 54, 34, 76};
  std::vector<int> out(in.size(), 0);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  solovyev_d_shell_sort_simple_tbb::TaskTBB task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), true);
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();
  ASSERT_TRUE(IsSorted(out));
}

TEST(solovyev_d_shell_sort_simple_tbb, sort_30_negative) {
  // Create data
  std::vector<int> in = {1,   5,   7, 3,  7,  3,  -8,   4,  1,   6,   4,  6, 7,   3, -12, 21,
                         -65, -43, 1, 54, 34, 76, -345, 21, 765, 346, 34, 1, 434, 8, 343, -88};
  std::vector<int> out(in.size(), 0);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  solovyev_d_shell_sort_simple_tbb::TaskTBB task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), true);
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();
  ASSERT_TRUE(IsSorted(out));
}

TEST(solovyev_d_shell_sort_simple_tbb, sort_30) {
  // Create data
  std::vector<int> in = {1,  5,  7, 3,  7,  3,  8,   4,  1,   6,   4,  6, 7,   3, 12,  21,
                         65, 43, 1, 54, 34, 76, 345, 21, 765, 346, 34, 1, 434, 8, 343, 88};
  std::vector<int> out(in.size(), 0);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  solovyev_d_shell_sort_simple_tbb::TaskTBB task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), true);
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();
  ASSERT_TRUE(IsSorted(out));
}

TEST(solovyev_d_shell_sort_simple_tbb, sort_rand_10) {
  // Create data
  std::vector<int> in = GetRandomVector(10);
  std::vector<int> out(in.size(), 0);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  solovyev_d_shell_sort_simple_tbb::TaskTBB task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), true);
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();
  ASSERT_TRUE(IsSorted(out));
}

TEST(solovyev_d_shell_sort_simple_tbb, sort_rand_100) {
  // Create data
  std::vector<int> in = GetRandomVector(100);
  std::vector<int> out(in.size(), 0);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  solovyev_d_shell_sort_simple_tbb::TaskTBB task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), true);
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();
  ASSERT_TRUE(IsSorted(out));
}