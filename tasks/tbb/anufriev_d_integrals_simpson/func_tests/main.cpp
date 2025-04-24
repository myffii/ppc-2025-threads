#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/anufriev_d_integrals_simpson/include/ops_tbb.hpp"

namespace {
const double kPi = std::numbers::pi;

std::shared_ptr<ppc::core::TaskData> MakeTaskData(const std::vector<double>& elements,
                                                  std::vector<double>& out_buffer) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  auto* input_ptr = reinterpret_cast<uint8_t*>(const_cast<double*>(elements.data()));
  auto* output_ptr = reinterpret_cast<uint8_t*>(out_buffer.data());
  task_data->inputs.push_back(input_ptr);
  task_data->inputs_count.push_back(static_cast<uint32_t>(elements.size() * sizeof(double)));
  task_data->outputs.push_back(output_ptr);
  task_data->outputs_count.push_back(static_cast<uint32_t>(out_buffer.size() * sizeof(double)));
  return task_data;
}
}  // namespace

TEST(anufriev_d_integrals_simpson_tbb, test_1D_sin) {
  std::vector<double> in = {1, 0.0, kPi / 2.0, 100, 1};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  double result = out_buffer[0];
  EXPECT_NEAR(result, 1.0, 1e-3);
}

TEST(anufriev_d_integrals_simpson_tbb, test_2D_sum_of_squares) {
  std::vector<double> in = {2, 0.0, 1.0, 100, 0.0, 1.0, 100, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  double result = out_buffer[0];
  EXPECT_NEAR(result, 2.0 / 3.0, 1e-3);
}

TEST(anufriev_d_integrals_simpson_tbb, test_2D_sin_cos) {
  std::vector<double> in = {2, 0.0, kPi / 2.0, 200, 0.0, kPi / 2.0, 200, 1};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  double result = out_buffer[0];
  EXPECT_NEAR(result, 1.0, 1e-3);
}

TEST(anufriev_d_integrals_simpson_tbb, test_unknown_func) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 999};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  double result = out_buffer[0];
  EXPECT_DOUBLE_EQ(result, 0.0);
}

TEST(anufriev_d_integrals_simpson_tbb, test_invalid_empty_input_ptr) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(nullptr);
  task_data->inputs_count.push_back(0);
  std::vector<double> out_buffer(1, 0.0);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_buffer.data()));
  task_data->outputs_count.push_back(sizeof(double));

  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(task_data);
  ASSERT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_tbb, test_invalid_output_buffer) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<double> in = {1, 0.0, 1.0, 2, 0};
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.push_back(in.size() * sizeof(double));
  task_data->outputs.push_back(nullptr);
  task_data->outputs_count.push_back(0);

  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(task_data);
  ASSERT_FALSE(task.ValidationImpl());
}

TEST(anufriev_d_integrals_simpson_tbb, test_invalid_dimension_zero) {
  std::vector<double> in = {0, 0.0, 1.0, 2, 999};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_tbb, test_invalid_dimension_negative) {
  std::vector<double> in = {-1, 0.0, 1.0, 2, 999};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_tbb, test_invalid_not_enough_data) {
  std::vector<double> in = {2, 0.0, 1.0, 200};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_tbb, test_invalid_odd_n) {
  std::vector<double> in = {1, 0.0, 1.0, 3, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_tbb, test_invalid_negative_n) {
  std::vector<double> in = {1, 0.0, 1.0, -2, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_tbb, test_no_output_buffer_in_taskdata) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 0};
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(in.data())));
  td->inputs_count.push_back(static_cast<std::uint32_t>(in.size() * sizeof(double)));
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  EXPECT_FALSE(task.Validation());
}

TEST(anufriev_d_integrals_simpson_tbb, test_invalid_n_not_integer) {
  std::vector<double> in = {1, 0.0, 1.0, 100.5, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_tbb, test_invalid_n_too_large) {
  std::vector<double> in = {1, 0.0, 1.0, static_cast<double>(std::numeric_limits<int>::max()) + 10.0, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_tbb, test_invalid_func_code_not_integer) {
  std::vector<double> in = {1, 0.0, 1.0, 100, 1.5};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_tbb, test_invalid_func_code_too_large) {
  std::vector<double> in = {1, 0.0, 1.0, 100, static_cast<double>(std::numeric_limits<int>::max()) + 10.0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_tbb, test_invalid_func_code_too_small) {
  std::vector<double> in = {1, 0.0, 1.0, 100, static_cast<double>(std::numeric_limits<int>::min()) - 10.0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_tbb, test_a_greater_than_b) {
  std::vector<double> in = {1, 1.0, 0.0, 100, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  double result = out_buffer[0];
  EXPECT_NEAR(result, -1.0 / 3.0, 1e-3);
}

TEST(anufriev_d_integrals_simpson_tbb, test_invalid_empty_output_count) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 0};
  std::vector<double> out_buf(1);
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(in.data())));
  td->inputs_count.push_back(static_cast<std::uint32_t>(in.size() * sizeof(double)));
  td->outputs.push_back(reinterpret_cast<uint8_t*>(out_buf.data()));
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(anufriev_d_integrals_simpson_tbb, test_invalid_small_output_count) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 0};
  std::vector<double> out_buf(1);
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(in.data())));
  td->inputs_count.push_back(static_cast<std::uint32_t>(in.size() * sizeof(double)));
  td->outputs.push_back(reinterpret_cast<uint8_t*>(out_buf.data()));
  td->outputs_count.push_back(sizeof(double) - 1);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  EXPECT_FALSE(task.ValidationImpl());
}