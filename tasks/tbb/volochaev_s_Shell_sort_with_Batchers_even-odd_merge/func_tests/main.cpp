#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/volochaev_s_Shell_sort_with_Batchers_even-odd_merge/include/ops_tbb.hpp"

namespace {
void GetRandomVector(std::vector<int> &v, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());

  if (a >= b) {
    throw std::invalid_argument("error.");
  }

  std::uniform_int_distribution<> dis(a, b);

  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = dis(gen);
  }
}
}  // namespace

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_error_in_val) {
  constexpr size_t kSizeOfVector = 0;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  std::vector<int> out(kSizeOfVector, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_error_in_generate) {
  constexpr size_t kSizeOfVector = 100;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  ASSERT_ANY_THROW(GetRandomVector(in, 1000, -1000));
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_small_vector) {
  constexpr size_t kSizeOfVector = 100;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_small_vector2) {
  constexpr size_t kSizeOfVector = 200;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_small_vector3) {
  constexpr size_t kSizeOfVector = 300;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_small_vector4) {
  constexpr size_t kSizeOfVector = 400;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_medium_vector) {
  constexpr size_t kSizeOfVector = 500;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_medium_vector2) {
  constexpr size_t kSizeOfVector = 600;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_medium_vector3) {
  constexpr size_t kSizeOfVector = 700;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_medium_vector4) {
  constexpr size_t kSizeOfVector = 800;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_medium_vector5) {
  constexpr size_t kSizeOfVector = 900;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_big_vector) {
  constexpr size_t kSizeOfVector = 1000;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_big_vector2) {
  constexpr size_t kSizeOfVector = 2000;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_big_vector3) {
  constexpr size_t kSizeOfVector = 3000;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_big_vector4) {
  constexpr size_t kSizeOfVector = 4000;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_extra_big_vector) {
  constexpr size_t kSizeOfVector = 10000;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_prime_size_vector) {
  constexpr size_t kSizeOfVector = 7;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_prime_size_vector1) {
  constexpr size_t kSizeOfVector = 13;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_prime_size_vector2) {
  constexpr size_t kSizeOfVector = 17;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_prime_size_vector3) {
  constexpr size_t kSizeOfVector = 23;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_prime_size_vector4) {
  constexpr size_t kSizeOfVector = 29;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_odd_number_of_elements) {
  constexpr size_t kSizeOfVector = 101;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_odd_number_of_elements1) {
  constexpr size_t kSizeOfVector = 99;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_odd_number_of_elements2) {
  constexpr size_t kSizeOfVector = 201;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_odd_number_of_elements3) {
  constexpr size_t kSizeOfVector = 199;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_odd_number_of_elements4) {
  constexpr size_t kSizeOfVector = 301;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_odd_number_of_elements5) {
  constexpr size_t kSizeOfVector = 299;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_odd_number_of_elements6) {
  constexpr size_t kSizeOfVector = 401;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_odd_number_of_elements7) {
  constexpr size_t kSizeOfVector = 399;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb, test_with_reverse) {
  constexpr size_t kSizeOfVector = 399;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  std::ranges::sort(in);
  std::ranges::reverse(in);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Fermats_1) {
  constexpr size_t kSizeOfVector = 3;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Fermats_2) {
  constexpr size_t kSizeOfVector = 5;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Fermats_3) {
  constexpr size_t kSizeOfVector = 17;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Fermats_4) {
  constexpr size_t kSizeOfVector = 257;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Fermats_5) {
  constexpr size_t kSizeOfVector = 65537;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_2_1) {
  constexpr size_t kSizeOfVector = 561;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_2_2) {
  constexpr size_t kSizeOfVector = 1105;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_2_3) {
  constexpr size_t kSizeOfVector = 1729;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_2_4) {
  constexpr size_t kSizeOfVector = 1905;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_2_5) {
  constexpr size_t kSizeOfVector = 2047;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_2_6) {
  constexpr size_t kSizeOfVector = 2465;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_2_7) {
  constexpr size_t kSizeOfVector = 3277;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_2_8) {
  constexpr size_t kSizeOfVector = 4033;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_2_9) {
  constexpr size_t kSizeOfVector = 4681;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_2_10) {
  constexpr size_t kSizeOfVector = 6601;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_3_1) {
  constexpr size_t kSizeOfVector = 121;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_3_2) {
  constexpr size_t kSizeOfVector = 703;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_3_3) {
  constexpr size_t kSizeOfVector = 1729;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_3_4) {
  constexpr size_t kSizeOfVector = 2821;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_3_5) {
  constexpr size_t kSizeOfVector = 3281;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_3_6) {
  constexpr size_t kSizeOfVector = 7381;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_3_7) {
  constexpr size_t kSizeOfVector = 8401;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_3_8) {
  constexpr size_t kSizeOfVector = 8911;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_3_9) {
  constexpr size_t kSizeOfVector = 10585;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Euler_base_3_10) {
  constexpr size_t kSizeOfVector = 12403;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Mersenne_1) {
  constexpr size_t kSizeOfVector = 16383;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Mersenne_2) {
  constexpr size_t kSizeOfVector = 32767;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Mersenne_3) {
  constexpr size_t kSizeOfVector = 65535;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Mersenne_4) {
  constexpr size_t kSizeOfVector = 131071;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_tbb, test_with_len_Mersenne_5) {
  constexpr size_t kSizeOfVector = 524287;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}
