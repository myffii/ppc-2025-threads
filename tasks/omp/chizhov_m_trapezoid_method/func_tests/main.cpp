#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/chizhov_m_trapezoid_method/include/ops_omp.hpp"

namespace {
void RunTests(int div, int dimm, std::vector<double> &limits, std::function<double(const std::vector<double> &)> f,
              double expected_result) {
  std::vector<double> res(1, 0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(std::move(f));

  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_omp->inputs_count.emplace_back(sizeof(div));

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dimm));
  task_data_omp->inputs_count.emplace_back(sizeof(dimm));

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_omp->inputs_count.emplace_back(limits.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_omp->outputs_count.emplace_back(res.size() * sizeof(double));

  chizhov_m_trapezoid_method_omp::TestTaskOpenMP test_task_omp(task_data_omp);

  ASSERT_TRUE(test_task_omp.ValidationImpl());
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  ASSERT_NEAR(res[0], expected_result, 0.1);
  delete f_object;
}

TEST(chizhov_m_trapezoid_method_omp, one_variable_squared) {
  int div = 20;
  int dim = 1;
  std::vector<double> limits = {0.0, 5.0};

  auto f = [](const std::vector<double> &f_val) { return f_val[0] * f_val[0]; };
  RunTests(div, dim, limits, f, 41.66);
}

TEST(chizhov_m_trapezoid_method_omp, one_variable_cube) {
  int div = 45;
  int dim = 1;
  std::vector<double> limits = {0.0, 5.0};

  auto f = [](const std::vector<double> &f_val) { return f_val[0] * f_val[0] * f_val[0]; };
  RunTests(div, dim, limits, f, 156.25);
}

TEST(chizhov_m_trapezoid_method_omp, mul_two_variables) {
  int div = 10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  auto f = [](const std::vector<double> &f_val) { return f_val[0] * f_val[1]; };
  RunTests(div, dim, limits, f, 56.25);
}

TEST(chizhov_m_trapezoid_method_omp, sum_two_variables) {
  int div = 10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  auto f = [](const std::vector<double> &f_val) { return f_val[0] + f_val[1]; };
  RunTests(div, dim, limits, f, 60);
}

TEST(chizhov_m_trapezoid_method_omp, dif_two_variables) {
  int div = 10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  auto f = [](const std::vector<double> &f_val) { return f_val[1] - f_val[0]; };
  RunTests(div, dim, limits, f, -15);
}

TEST(chizhov_m_trapezoid_method_omp, cos_one_variable) {
  int div = 45;
  int dim = 1;
  std::vector<double> limits = {0.0, 5.0};

  auto f = [](const std::vector<double> &f_val) { return std::cos(f_val[0]); };
  RunTests(div, dim, limits, f, -0.95);
}

TEST(chizhov_m_trapezoid_method_seq, sin_two_variables) {
  int div = 45;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 5.0};

  auto f = [](const std::vector<double> &f_val) { return std::sin(f_val[0] + f_val[1]); };
  RunTests(div, dim, limits, f, -1.37);
}

TEST(chizhov_m_trapezoid_method_omp, exp_two_variables) {
  int div = 80;
  int dim = 2;
  std::vector<double> limits = {0.0, 3.0, 0.0, 3.0};

  auto f = [](const std::vector<double> &f_val) { return std::exp(f_val[0] + f_val[1]); };
  RunTests(div, dim, limits, f, 364.25);
}

TEST(chizhov_m_trapezoid_method_omp, combine_exp_sin_cos) {
  int div = 90;
  int dim = 2;
  std::vector<double> limits = {0.0, 3.0, 0.0, 3.0};

  auto f = [](const std::vector<double> &f_val) {
    return std::exp(-f_val[0]) * std::sin(f_val[0]) * std::cos(f_val[1]);
  };
  RunTests(div, dim, limits, f, 0.073);
}

}  // namespace

TEST(chizhov_m_trapezoid_method_omp, invalid_value_dim) {
  int div = 10;
  int dim = -2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_omp->inputs_count.emplace_back(sizeof(div));

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_omp->inputs_count.emplace_back(sizeof(dim));

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_omp->inputs_count.emplace_back(limits.size());

  chizhov_m_trapezoid_method_omp::TestTaskOpenMP test_task_omp(task_data_omp);

  ASSERT_FALSE(test_task_omp.ValidationImpl());
}

TEST(chizhov_m_trapezoid_method_omp, invalid_value_div) {
  int div = -10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_omp->inputs_count.emplace_back(sizeof(div));

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_omp->inputs_count.emplace_back(sizeof(dim));

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_omp->inputs_count.emplace_back(limits.size());

  chizhov_m_trapezoid_method_omp::TestTaskOpenMP test_task_omp(task_data_omp);

  ASSERT_FALSE(test_task_omp.ValidationImpl());
}

TEST(chizhov_m_trapezoid_method_omp, invalid_limit_size) {
  int div = -10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0};

  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_omp->inputs_count.emplace_back(sizeof(div));

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_omp->inputs_count.emplace_back(sizeof(dim));

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_omp->inputs_count.emplace_back(limits.size());

  chizhov_m_trapezoid_method_omp::TestTaskOpenMP test_task_omp(task_data_omp);

  ASSERT_FALSE(test_task_omp.ValidationImpl());
}
