#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/odintsov_m_multmatrix_cannon/include/ops_stl.hpp"

namespace odintsov_m_mulmatrix_cannon_stl {
namespace {
static std::vector<double> GenerateMatrix(int sz);
static std::vector<double> MultiplyMatrices(const std::vector<double> &a, const std::vector<double> &b, int n);

std::vector<double> GenerateMatrix(int sz) {
  std::vector<double> matrix(sz * sz);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-100.0, 100.0);
  for (int i = 0; i < sz; ++i) {
    for (int j = 0; j < sz; ++j) {
      matrix[(i * sz) + j] = dis(gen);
    }
  }
  return matrix;
}

std::vector<double> MultiplyMatrices(const std::vector<double> &a, const std::vector<double> &b, int n) {
  std::vector<double> c(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += a[(i * n) + k] * b[(k * n) + j];
      }
      c[(i * n) + j] = sum;
    }
  }
  return c;
}
}  // namespace
}  // namespace odintsov_m_mulmatrix_cannon_stl

TEST(odintsov_m_mulmatrix_cannon_stl, test_matrix_4) {
  std::vector<double> matrix_a = odintsov_m_mulmatrix_cannon_stl::GenerateMatrix(4);
  std::vector<double> matrix_b = odintsov_m_mulmatrix_cannon_stl::GenerateMatrix(4);
  std::vector<double> out_stl(16, 0);
  std::vector<double> out_ans = odintsov_m_mulmatrix_cannon_stl::MultiplyMatrices(matrix_a, matrix_b, 4);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_stl->inputs_count.emplace_back(matrix_a.size());
  task_data_stl->inputs_count.emplace_back(matrix_b.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_stl.data()));

  odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  ASSERT_EQ(out_ans.size(), out_stl.size());
  for (size_t i = 0; i < out_ans.size(); ++i) {
    EXPECT_NEAR(out_ans[i], out_stl[i], 1e-9);
  }
}

TEST(odintsov_m_mulmatrix_cannon_stl, test_matrix_100) {
  std::vector<double> matrix_a = odintsov_m_mulmatrix_cannon_stl::GenerateMatrix(10);
  std::vector<double> matrix_b = odintsov_m_mulmatrix_cannon_stl::GenerateMatrix(10);
  std::vector<double> out_stl(100, 0);
  std::vector<double> out_ans = odintsov_m_mulmatrix_cannon_stl::MultiplyMatrices(matrix_a, matrix_b, 10);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_stl->inputs_count.emplace_back(matrix_a.size());
  task_data_stl->inputs_count.emplace_back(matrix_b.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_stl.data()));

  odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  ASSERT_EQ(out_ans.size(), out_stl.size());
  for (size_t i = 0; i < out_ans.size(); ++i) {
    EXPECT_NEAR(out_ans[i], out_stl[i], 1e-9);
  }
}

TEST(odintsov_m_mulmatrix_cannon_stl, test_matrix_900) {
  std::vector<double> matrix_a = odintsov_m_mulmatrix_cannon_stl::GenerateMatrix(30);
  std::vector<double> matrix_b = odintsov_m_mulmatrix_cannon_stl::GenerateMatrix(30);
  std::vector<double> out_stl(900, 0);
  std::vector<double> out_ans = odintsov_m_mulmatrix_cannon_stl::MultiplyMatrices(matrix_a, matrix_b, 30);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_stl->inputs_count.emplace_back(matrix_a.size());
  task_data_stl->inputs_count.emplace_back(matrix_b.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_stl.data()));

  odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  ASSERT_EQ(out_ans.size(), out_stl.size());
  for (size_t i = 0; i < out_ans.size(); ++i) {
    EXPECT_NEAR(out_ans[i], out_stl[i], 1e-9);
  }
}

TEST(odintsov_m_mulmatrix_cannon_stl, test_block_sz_1) {
  std::vector<double> matrix_a = odintsov_m_mulmatrix_cannon_stl::GenerateMatrix(3);
  std::vector<double> matrix_b = odintsov_m_mulmatrix_cannon_stl::GenerateMatrix(3);
  std::vector<double> out_stl(9, 0);
  std::vector<double> out_ans = odintsov_m_mulmatrix_cannon_stl::MultiplyMatrices(matrix_a, matrix_b, 3);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_stl->inputs_count.emplace_back(matrix_a.size());
  task_data_stl->inputs_count.emplace_back(matrix_b.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_stl.data()));

  odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  ASSERT_EQ(out_ans.size(), out_stl.size());
  for (size_t i = 0; i < out_ans.size(); ++i) {
    EXPECT_NEAR(out_ans[i], out_stl[i], 1e-9);
  }
}

TEST(odintsov_m_mulmatrix_cannon_stl, test_validation) {
  std::vector<double> matrix_a(12, 0);
  std::vector<double> matrix_b(12, 0);
  std::vector<double> out_stl(12, 0);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_stl->inputs_count.emplace_back(matrix_a.size());
  task_data_stl->inputs_count.emplace_back(matrix_b.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_stl.data()));

  odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), false);
}
