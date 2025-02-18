#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"

namespace {
std::vector<double> GenerateRandomMatrix(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(0.0, 100.0);

  std::vector<double> matrix(size * size);
  for (size_t i = 0; i < size * size; ++i) {
    matrix[i] = distrib(gen);
  }
  return matrix;
}
}  // namespace

TEST(nasedkin_e_strassen_algorithm_seq, test_matrix_63x63_fixed) {
  constexpr size_t kMatrixSize = 63;

  std::vector<double> in_a(kMatrixSize * kMatrixSize);
  std::vector<double> in_b(kMatrixSize * kMatrixSize);

  for (size_t i = 0; i < kMatrixSize * kMatrixSize; ++i) {
    in_a[i] = static_cast<double>((kMatrixSize * kMatrixSize) - i);
  }

  for (size_t i = 0; i < kMatrixSize * kMatrixSize; ++i) {
    in_b[i] = static_cast<double>(i + 1);
  }

  std::vector<double> expected_result(kMatrixSize * kMatrixSize);

  for (size_t i = 0; i < kMatrixSize; ++i) {
    for (size_t j = 0; j < kMatrixSize; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < kMatrixSize; ++k) {
        sum += in_a[(i * kMatrixSize) + k] * in_b[(k * kMatrixSize) + j];
      }
      expected_result[(i * kMatrixSize) + j] = sum;
    }
  }

  std::vector<double> out(kMatrixSize * kMatrixSize, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  nasedkin_e_strassen_algorithm_seq::StrassenSequential strassen_task_sequential(task_data_seq);
  ASSERT_EQ(strassen_task_sequential.Validation(), true);
  strassen_task_sequential.PreProcessing();
  strassen_task_sequential.Run();
  strassen_task_sequential.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_NEAR(expected_result[i], out[i], 1e-6);
  }
}

TEST(nasedkin_e_strassen_algorithm_seq, test_matrix_64x64_fixed) {
  constexpr size_t kMatrixSize = 64;

  std::vector<double> in_a(kMatrixSize * kMatrixSize);
  std::vector<double> in_b(kMatrixSize * kMatrixSize);

  for (size_t i = 0; i < kMatrixSize * kMatrixSize; ++i) {
    in_a[i] = static_cast<double>((kMatrixSize * kMatrixSize) - i);
  }

  for (size_t i = 0; i < kMatrixSize * kMatrixSize; ++i) {
    in_b[i] = static_cast<double>(i + 1);
  }

  std::vector<double> expected_result(kMatrixSize * kMatrixSize);

  for (size_t i = 0; i < kMatrixSize; ++i) {
    for (size_t j = 0; j < kMatrixSize; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < kMatrixSize; ++k) {
        sum += in_a[(i * kMatrixSize) + k] * in_b[(k * kMatrixSize) + j];
      }
      expected_result[(i * kMatrixSize) + j] = sum;
    }
  }

  std::vector<double> out(kMatrixSize * kMatrixSize, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  nasedkin_e_strassen_algorithm_seq::StrassenSequential strassen_task_sequential(task_data_seq);
  ASSERT_EQ(strassen_task_sequential.Validation(), true);
  strassen_task_sequential.PreProcessing();
  strassen_task_sequential.Run();
  strassen_task_sequential.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_NEAR(expected_result[i], out[i], 1e-6);
  }
}

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_64x64) {
  constexpr size_t kMatrixSize = 64;

  std::vector<double> in_a = GenerateRandomMatrix(kMatrixSize);
  std::vector<double> in_b = GenerateRandomMatrix(kMatrixSize);
  std::vector<double> out(kMatrixSize * kMatrixSize, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  nasedkin_e_strassen_algorithm_seq::StrassenSequential strassen_task_sequential(task_data_seq);
  ASSERT_EQ(strassen_task_sequential.Validation(), true);
  strassen_task_sequential.PreProcessing();
  strassen_task_sequential.Run();
  strassen_task_sequential.PostProcessing();
}

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_127x127) {
  constexpr size_t kMatrixSize = 128;

  std::vector<double> in_a = GenerateRandomMatrix(kMatrixSize);
  std::vector<double> in_b = GenerateRandomMatrix(kMatrixSize);
  std::vector<double> out(kMatrixSize * kMatrixSize, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  nasedkin_e_strassen_algorithm_seq::StrassenSequential strassen_task_sequential(task_data_seq);
  ASSERT_EQ(strassen_task_sequential.Validation(), true);
  strassen_task_sequential.PreProcessing();
  strassen_task_sequential.Run();
  strassen_task_sequential.PostProcessing();
}

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_128x128) {
  constexpr size_t kMatrixSize = 128;

  std::vector<double> in_a = GenerateRandomMatrix(kMatrixSize);
  std::vector<double> in_b = GenerateRandomMatrix(kMatrixSize);
  std::vector<double> out(kMatrixSize * kMatrixSize, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  nasedkin_e_strassen_algorithm_seq::StrassenSequential strassen_task_sequential(task_data_seq);
  ASSERT_EQ(strassen_task_sequential.Validation(), true);
  strassen_task_sequential.PreProcessing();
  strassen_task_sequential.Run();
  strassen_task_sequential.PostProcessing();
}

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_255x255) {
  constexpr size_t kMatrixSize = 256;

  std::vector<double> in_a = GenerateRandomMatrix(kMatrixSize);
  std::vector<double> in_b = GenerateRandomMatrix(kMatrixSize);
  std::vector<double> out(kMatrixSize * kMatrixSize, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  nasedkin_e_strassen_algorithm_seq::StrassenSequential strassen_task_sequential(task_data_seq);
  ASSERT_EQ(strassen_task_sequential.Validation(), true);
  strassen_task_sequential.PreProcessing();
  strassen_task_sequential.Run();
  strassen_task_sequential.PostProcessing();
}

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_256x256) {
  constexpr size_t kMatrixSize = 256;

  std::vector<double> in_a = GenerateRandomMatrix(kMatrixSize);
  std::vector<double> in_b = GenerateRandomMatrix(kMatrixSize);
  std::vector<double> out(kMatrixSize * kMatrixSize, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  nasedkin_e_strassen_algorithm_seq::StrassenSequential strassen_task_sequential(task_data_seq);
  ASSERT_EQ(strassen_task_sequential.Validation(), true);
  strassen_task_sequential.PreProcessing();
  strassen_task_sequential.Run();
  strassen_task_sequential.PostProcessing();
}