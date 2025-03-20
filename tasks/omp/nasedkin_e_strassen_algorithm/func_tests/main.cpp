#include <gtest/gtest.h>
#include <vector>
#include <random>
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

  std::vector<double> expected = nasedkin_e_strassen_algorithm_omp::StandardMultiply(in_a, in_b, size);
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
}

TEST(nasedkin_e_strassen_algorithm_omp, test_matrix_64x64_fixed) {
  RunFixedMatrixTest(64);
}

TEST(nasedkin_e_strassen_algorithm_omp, test_matrix_128x128_random) {
  RunRandomMatrixTest(128);
}

TEST(nasedkin_e_strassen_algorithm_omp, test_matrix_256x256_random) {
  RunRandomMatrixTest(256);
}