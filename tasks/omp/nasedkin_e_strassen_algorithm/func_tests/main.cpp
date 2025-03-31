#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/nasedkin_e_strassen_algorithm/include/ops_omp.hpp"

namespace {
std::vector<double> GenerateRandomMatrix(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(-100.0, 100.0);
  std::vector<double> matrix(size * size);
  for (int i = 0; i < size * size; ++i) {
    matrix[i] = distrib(gen);
  }
  return matrix;
}

void RunRandomMatrixTest(int size) {
  std::vector<double> in_a = GenerateRandomMatrix(size);
  std::vector<double> in_b = GenerateRandomMatrix(size);
  std::vector<double> out(size * size, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data->inputs_count.emplace_back(in_a.size());
  task_data->inputs_count.emplace_back(in_b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  nasedkin_e_strassen_algorithm_omp::StrassenOmp strassen_task(task_data);
  ASSERT_TRUE(strassen_task.Validation());
  strassen_task.PreProcessing();
  strassen_task.Run();
  strassen_task.PostProcessing();
}

void RunFixedMatrixTest(int size) {
  std::vector<double> in_a(size * size);
  std::vector<double> in_b(size * size);

  for (int i = 0; i < size * size; ++i) {
    in_a[i] = static_cast<double>((size * size) - i);
    in_b[i] = static_cast<double>(i + 1);
  }

  std::vector<double> expected =
      nasedkin_e_strassen_algorithm_omp::StandardMultiply(in_a, in_b, size);
  std::vector<double> out(size * size, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data->inputs_count.emplace_back(in_a.size());
  task_data->inputs_count.emplace_back(in_b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  nasedkin_e_strassen_algorithm_omp::StrassenOmp strassen_task(task_data);
  ASSERT_TRUE(strassen_task.Validation());
  strassen_task.PreProcessing();
  strassen_task.Run();
  strassen_task.PostProcessing();

  for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
    EXPECT_NEAR(expected[i], out[i], 1e-6);
  }
}
}  // namespace

TEST(nasedkin_e_strassen_algorithm_omp, test_matrix_64x32_32x64_fixed) {
  const int rows_a = 64, cols_a = 32, cols_b = 64;

  std::vector<double> in_a(rows_a * cols_a);
  std::vector<double> in_b(cols_a * cols_b);

  for (int i = 0; i < rows_a * cols_a; ++i) {
    in_a[i] = static_cast<double>((rows_a * cols_a) - i);
  }
  for (int i = 0; i < cols_a * cols_b; ++i) {
    in_b[i] = static_cast<double>(i + 1);
  }

  std::vector<double> expected = nasedkin_e_strassen_algorithm_omp::StandardMultiply(in_a, in_b, rows_a, cols_a, cols_b);
  std::vector<double> out(rows_a * cols_b, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data->inputs_count.emplace_back(in_a.size());
  task_data->inputs_count.emplace_back(in_b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  nasedkin_e_strassen_algorithm_omp::StrassenOmp strassen_task(task_data);
  ASSERT_TRUE(strassen_task.Validation());
  strassen_task.PreProcessing();
  strassen_task.Run();
  strassen_task.PostProcessing();

  for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
    EXPECT_NEAR(expected[i], out[i], 1e-6);
  }
}

TEST(nasedkin_e_strassen_algorithm_omp, test_matrix_63x63_fixed) { RunFixedMatrixTest(63); }

TEST(nasedkin_e_strassen_algorithm_omp, test_matrix_64x64_fixed) { RunFixedMatrixTest(64); }

TEST(nasedkin_e_strassen_algorithm_omp, test_matrix_32x64_64x32_random) {
  const int rows_a = 32, cols_a = 64, cols_b = 32;

  std::vector<double> in_a = GenerateRandomMatrix(rows_a * cols_a);
  std::vector<double> in_b = GenerateRandomMatrix(cols_a * cols_b);
  std::vector<double> out(rows_a * cols_b, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data->inputs_count.emplace_back(in_a.size());
  task_data->inputs_count.emplace_back(in_b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  nasedkin_e_strassen_algorithm_omp::StrassenOmp strassen_task(task_data);
  ASSERT_TRUE(strassen_task.Validation());
  strassen_task.PreProcessing();
  strassen_task.Run();
  strassen_task.PostProcessing();
}

TEST(nasedkin_e_strassen_algorithm_omp, test_matrix_64x64_random) { RunRandomMatrixTest(64); }

TEST(nasedkin_e_strassen_algorithm_omp, test_matrix_127x127_random) { RunRandomMatrixTest(127); }

TEST(nasedkin_e_strassen_algorithm_omp, test_matrix_128x128_random) { RunRandomMatrixTest(128); }

TEST(nasedkin_e_strassen_algorithm_omp, test_matrix_255x255_random) { RunRandomMatrixTest(255); }

TEST(nasedkin_e_strassen_algorithm_omp, test_matrix_256x256_random) { RunRandomMatrixTest(256); }