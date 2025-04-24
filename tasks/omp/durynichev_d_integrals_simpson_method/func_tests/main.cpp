#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/durynichev_d_integrals_simpson_method/include/ops_omp.hpp"

// 1D tests
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_1D_x_squared) {
  std::vector<double> in = {0.0, 1.0, 100, 0};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  EXPECT_NEAR(out[0], 1.0 / 3.0, 1e-5);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_1D_sin) {
  std::vector<double> in = {0.0, std::numbers::pi, 1000, 1};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  // 2.0
  EXPECT_NEAR(out[0], 2.0, 1e-4);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_1D_cos) {
  std::vector<double> in = {0.0, std::numbers::pi / 2, 1000, 2};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  // 1.0
  EXPECT_NEAR(out[0], 1.0, 1e-4);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_1D_exp) {
  std::vector<double> in = {0.0, 1.0, 1000, 3};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  // e-1 ≈ 1.718
  EXPECT_NEAR(out[0], std::exp(1.0) - 1.0, 1e-4);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_1D_log) {
  std::vector<double> in = {1.0, std::numbers::e, 1000, 4};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  // 1.0
  EXPECT_NEAR(out[0], 1.0, 1e-4);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_1D_combined) {
  std::vector<double> in = {0.0, 1.0, 1000, 5};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  // Analytical result: (sin(1) - sin(0)) + (sin(1) - 0) + 1/3
  double expected = std::sin(1.0) + (1.0 - std::cos(1.0)) + (1.0 / 3.0);
  EXPECT_NEAR(out[0], expected, 1e-4);
}

// 2D
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_2D_sin) {
  std::vector<double> in = {0.0, std::numbers::pi / 2, 0.0, std::numbers::pi / 2, 100, 1};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  // (1 - cos(π/2))^2 = 1
  EXPECT_NEAR(out[0], 1.0, 1e-4);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_2D_cos) {
  std::vector<double> in = {0.0, std::numbers::pi / 2, 0.0, std::numbers::pi / 2, 100, 2};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  // sin(π/2)*sin(π/2) = 1
  EXPECT_NEAR(out[0], 1.0, 1e-4);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_2D_exp) {
  std::vector<double> in = {0.0, 1.0, 0.0, 1.0, 100, 3};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  // Analytical result: (e-1)^2
  double expected = (std::numbers::e - 1.0) * (std::numbers::e - 1.0);
  EXPECT_NEAR(out[0], expected, 1e-4);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_2D_combined) {
  std::vector<double> in = {0.0, 1.0, 0.0, 1.0, 100, 5};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  // (1-cos(1)) * 1 + sin(1) * 1 + 1/3 * 1 + 1 * 1/3
  double expected = (1.0 - std::cos(1.0)) + std::sin(1.0) + (1.0 / 3.0) + (1.0 / 3.0);
  EXPECT_NEAR(out[0], expected, 1e-4);
}

// 3D
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_3D_x_squared) {
  std::vector<double> in = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 100, 0};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  // 1.0
  EXPECT_NEAR(out[0], 1.0, 1e-5);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_3D_sin) {
  std::vector<double> in = {0.0, std::numbers::pi / 2, 0.0, std::numbers::pi / 2, 0.0, std::numbers::pi / 2, 100, 1};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  // (1-cos(π/2))^3 = 1
  EXPECT_NEAR(out[0], 1.0, 1e-4);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_3D_cos) {
  std::vector<double> in = {0.0, std::numbers::pi / 2, 0.0, std::numbers::pi / 2, 0.0, std::numbers::pi / 2, 100, 2};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  // sin(π/2)^3 = 1
  EXPECT_NEAR(out[0], 1.0, 1e-4);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_3D_exp) {
  std::vector<double> in = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 100, 3};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  // (e-1)^3
  double expected = std::pow(std::numbers::e - 1.0, 3);
  EXPECT_NEAR(out[0], expected, 1e-4);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_3D_combined) {
  std::vector<double> in = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 100, 5};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  // (1-cos(1)) + sin(1) + (1-cos(1)) + 1/3 + 1/3 + 1/3
  double expected = (1.0 - std::cos(1.0)) + std::sin(1.0) + (1.0 - std::cos(1.0)) + 1.0;
  EXPECT_NEAR(out[0], expected, 1e-4);
}

// Random tests
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_1D_random) {
  // Random bounds
  double a = -10.0 + ((rand() % 200) / 10.0);  // from -10 to 10
  double b = a + ((rand() % 100) / 10.0);      // from a to a+10
  int n = 100 + (rand() % 1000);               // from 100 to 1100 steps
  n = n - (n % 2);                             // Make n even

  std::vector<double> in = {a, b, static_cast<double>(n), 0};  // Quadratic function
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  // Analytical result for x^2 on the interval [a,b]
  double expected = (b * b * b - a * a * a) / 3.0;
  EXPECT_NEAR(out[0], expected, 1e-3);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_2D_random) {
  // Random bounds
  double x_a = -5.0 + ((rand() % 100) / 10.0);  // from -5 to 5
  double x_b = x_a + ((rand() % 50) / 10.0);    // from x_a to x_a+5
  double y_a = -5.0 + ((rand() % 100) / 10.0);  // from -5 to 5
  double y_b = y_a + ((rand() % 50) / 10.0);    // from y_a to y_a+5
  int n = 100 + (rand() % 200);                 // from 100 to 300 steps
  n = n - (n % 2);                              // Make n even

  std::vector<double> in = {x_a, x_b, y_a, y_b, static_cast<double>(n), 0};  // Quadratic function
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  // Analytical result for x^2 + y^2 on the rectangle [x_a,x_b]x[y_a,y_b]
  double expected_x = (x_b * x_b * x_b - x_a * x_a * x_a) / 3.0;
  double expected_y = (y_b * y_b * y_b - y_a * y_a * y_a) / 3.0;
  double expected = (expected_x * (y_b - y_a)) + (expected_y * (x_b - x_a));
  EXPECT_NEAR(out[0], expected, 1e-3);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_3D_random) {
  // Random bounds
  double x_a = -2.0 + ((rand() % 40) / 10.0);  // from -2 to 2
  double x_b = x_a + ((rand() % 20) / 10.0);   // from x_a to x_a+2
  double y_a = -2.0 + ((rand() % 40) / 10.0);  // from -2 to 2
  double y_b = y_a + ((rand() % 20) / 10.0);   // from y_a to y_a+2
  double z_a = -2.0 + ((rand() % 40) / 10.0);  // from -2 to 2
  double z_b = z_a + ((rand() % 20) / 10.0);   // from z_a to z_a+2
  int n = 50 + (rand() % 100);                 // from 50 to 150 steps
  n = n - (n % 2);                             // Make n even

  std::vector<double> in = {x_a, x_b, y_a, y_b, z_a, z_b, static_cast<double>(n), 0};  // Quadratic function
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  // Analytical result for x^2 + y^2 + z^2 on the rectangular prism [x_a,x_b]x[y_a,y_b]x[z_a,z_b]
  double expected_x = (x_b * x_b * x_b - x_a * x_a * x_a) / 3.0;
  double expected_y = (y_b * y_b * y_b - y_a * y_a * y_a) / 3.0;
  double expected_z = (z_b * z_b * z_b - z_a * z_a * z_a) / 3.0;
  double expected = (expected_x * (y_b - y_a) * (z_b - z_a)) + (expected_y * (x_b - x_a) * (z_b - z_a)) +
                    (expected_z * (x_b - x_a) * (y_b - y_a));
  EXPECT_NEAR(out[0], expected, 1e-3);
}
