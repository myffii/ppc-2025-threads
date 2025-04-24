#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/odintsov_m_multmatrix_cannon/include/ops_omp.hpp"

namespace odintsov_m_mulmatrix_cannon_omp {

namespace {
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

}  // namespace odintsov_m_mulmatrix_cannon_omp

TEST(odintsov_m_mulmatrix_cannon_omp, test_matrix_4) {
  std::vector<double> matrix_a = odintsov_m_mulmatrix_cannon_omp::GenerateMatrix(4);
  std::vector<double> matrix_b = odintsov_m_mulmatrix_cannon_omp::GenerateMatrix(4);
  std::vector<double> out_omp(16, 0);
  std::vector<double> out_ans = odintsov_m_mulmatrix_cannon_omp::MultiplyMatrices(matrix_a, matrix_b, 4);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs_count.emplace_back(matrix_b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));

  odintsov_m_mulmatrix_cannon_omp::MulMatrixCannonOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  ASSERT_EQ(out_ans.size(), out_omp.size());
  for (size_t i = 0; i < out_ans.size(); ++i) {
    EXPECT_NEAR(out_ans[i], out_omp[i], 1e-9);
  }
}

TEST(odintsov_m_mulmatrix_cannon_omp, test_matrix_100) {
  std::vector<double> matrix_a = odintsov_m_mulmatrix_cannon_omp::GenerateMatrix(100);
  std::vector<double> matrix_b = odintsov_m_mulmatrix_cannon_omp::GenerateMatrix(100);
  std::vector<double> out_omp(10000, 0);
  std::vector<double> out_ans = odintsov_m_mulmatrix_cannon_omp::MultiplyMatrices(matrix_a, matrix_b, 100);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs_count.emplace_back(matrix_b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));

  odintsov_m_mulmatrix_cannon_omp::MulMatrixCannonOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  ASSERT_EQ(out_ans.size(), out_omp.size());
  for (size_t i = 0; i < out_ans.size(); ++i) {
    EXPECT_NEAR(out_ans[i], out_omp[i], 1e-9);
  }
}

TEST(odintsov_m_mulmatrix_cannon_omp, test_matrix_900) {
  std::vector<double> matrix_a = odintsov_m_mulmatrix_cannon_omp::GenerateMatrix(30);
  std::vector<double> matrix_b = odintsov_m_mulmatrix_cannon_omp::GenerateMatrix(30);
  std::vector<double> out_omp(900, 0);
  std::vector<double> out_ans = odintsov_m_mulmatrix_cannon_omp::MultiplyMatrices(matrix_a, matrix_b, 30);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs_count.emplace_back(matrix_b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));

  odintsov_m_mulmatrix_cannon_omp::MulMatrixCannonOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  ASSERT_EQ(out_ans.size(), out_omp.size());
  for (size_t i = 0; i < out_ans.size(); ++i) {
    EXPECT_NEAR(out_ans[i], out_omp[i], 1e-9);
  }
}

TEST(odintsov_m_mulmatrix_cannon_omp, test_sz_block_1) {
  std::vector<double> matrix_a = odintsov_m_mulmatrix_cannon_omp::GenerateMatrix(3);
  std::vector<double> matrix_b = odintsov_m_mulmatrix_cannon_omp::GenerateMatrix(3);
  std::vector<double> out_omp(9, 0);
  std::vector<double> out_ans = odintsov_m_mulmatrix_cannon_omp::MultiplyMatrices(matrix_a, matrix_b, 3);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs_count.emplace_back(matrix_b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));

  odintsov_m_mulmatrix_cannon_omp::MulMatrixCannonOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  ASSERT_EQ(out_ans.size(), out_omp.size());
  for (size_t i = 0; i < out_ans.size(); ++i) {
    EXPECT_NEAR(out_ans[i], out_omp[i], 1e-9);
  }
}

TEST(odintsov_m_mulmatrix_cannon_omp, test_validation) {
  std::vector<double> matrix_a(12, 0);
  std::vector<double> matrix_b(12, 0);
  std::vector<double> out_omp(12, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs_count.emplace_back(matrix_b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));

  odintsov_m_mulmatrix_cannon_omp::MulMatrixCannonOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), false);
}

TEST(odintsov_m_mulmatrix_cannon_omp, test_no_random_1) {
  std::vector<double> matrix_a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> matrix_b{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> out_omp(16, 0);
  std::vector<double> out_ans = odintsov_m_mulmatrix_cannon_omp::MultiplyMatrices(matrix_a, matrix_b, 4);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs_count.emplace_back(matrix_b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));

  odintsov_m_mulmatrix_cannon_omp::MulMatrixCannonOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  ASSERT_EQ(out_ans.size(), out_omp.size());
  for (size_t i = 0; i < out_ans.size(); ++i) {
    EXPECT_NEAR(out_ans[i], out_omp[i], 1e-9);
  }
}
TEST(odintsov_m_mulmatrix_cannon_omp, test_no_random_2) {
  std::vector<double> matrix_a{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> matrix_b{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> out_omp(9, 0);
  std::vector<double> out_ans = odintsov_m_mulmatrix_cannon_omp::MultiplyMatrices(matrix_a, matrix_b, 3);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs_count.emplace_back(matrix_b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));

  odintsov_m_mulmatrix_cannon_omp::MulMatrixCannonOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  ASSERT_EQ(out_ans.size(), out_omp.size());
  for (size_t i = 0; i < out_ans.size(); ++i) {
    EXPECT_NEAR(out_ans[i], out_omp[i], 1e-9);
  }
}