#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/kozlova_e_contrast_enhancement/include/ops_omp.hpp"

namespace {

std::vector<uint8_t> GenerateVector(int length) {
  std::vector<uint8_t> vec(length);
  for (int i = 0; i < length; ++i) {
    vec[i] = rand() % 256;
  }
  return vec;
}

std::shared_ptr<ppc::core::TaskData> CreateTaskData(std::vector<uint8_t>& in, std::vector<uint8_t>& out, size_t width,
                                                    size_t height) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(in.size());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  return task_data;
}

void TestRun(std::vector<uint8_t> in, std::vector<uint8_t> out, size_t width, size_t height) {
  auto task_data_omp = CreateTaskData(in, out, width, height);
  kozlova_e_contrast_enhancement_omp::TestTaskOpenMP test_task_omp(task_data_omp);

  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  uint8_t min_value = *std::ranges::min_element(in);
  uint8_t max_value = *std::ranges::max_element(in);

  for (size_t i = 0; i < in.size(); ++i) {
    uint8_t expected = (max_value == min_value)
                           ? in[i]
                           : static_cast<uint8_t>(((in[i] - min_value) / double(max_value - min_value)) * 255);
    EXPECT_EQ(out[i], expected);
  }
}

void TestValidation(std::vector<uint8_t> in, std::vector<uint8_t> out, size_t width, size_t height) {
  auto task_data_omp = CreateTaskData(in, out, width, height);
  kozlova_e_contrast_enhancement_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_FALSE(test_task_omp.Validation());
}

}  // namespace

TEST(kozlova_e_contrast_enhancement_omp, test_1st_image) {
  std::vector<uint8_t> in{10, 0, 50, 100, 200, 34};
  std::vector<uint8_t> out(6, 0);
  TestRun(in, out, 2, 3);
}

TEST(kozlova_e_contrast_enhancement_omp, test_large_image) {
  std::vector<uint8_t> in = GenerateVector(400);
  std::vector<uint8_t> out(400, 0);
  TestRun(in, out, 10, 40);
}

TEST(kozlova_e_contrast_enhancement_omp, test_empty_input) { TestValidation({}, {}, 0, 0); }

TEST(kozlova_e_contrast_enhancement_omp, test_same_values_input) {
  std::vector<uint8_t> in(6, 100);
  std::vector<uint8_t> out(6, 0);
  TestRun(in, out, 2, 3);
}

TEST(kozlova_e_contrast_enhancement_omp, test_difference_input) {
  std::vector<uint8_t> in{10, 20, 30, 100, 200, 250};
  std::vector<uint8_t> out(6, 0);
  TestRun(in, out, 2, 3);
}

TEST(kozlova_e_contrast_enhancement_omp, test_incorrect_input_size) { TestValidation({3, 3, 3}, {}, 3, 1); }

TEST(kozlova_e_contrast_enhancement_omp, test_incorrect_input_width) {
  TestValidation({3, 3, 3, 3}, {0, 0, 0, 0}, 3, 1);
}
