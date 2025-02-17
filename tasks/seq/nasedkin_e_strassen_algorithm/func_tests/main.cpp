#include <gtest/gtest.h>

#include <cstddef>  // Для size_t
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"

namespace {
// Метод для генерации случайной матрицы
std::vector<int> GenerateRandomMatrix(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, 100);

  std::vector<int> matrix(size * size);
  for (size_t i = 0; i < size * size; ++i) {
    matrix[i] = distrib(gen);
  }
  return matrix;
}
}  // namespace

TEST(nasedkin_e_strassen_algorithm_seq, test_matrix_64x64_fixed) {
  constexpr size_t kMatrixSize = 64;

  // Задаем фиксированные матрицы A и B размером 64x64
  std::vector<int> in_a(kMatrixSize * kMatrixSize);
  std::vector<int> in_b(kMatrixSize * kMatrixSize);

  // Заполняем матрицу A значениями от 4096 до 1
  for (size_t i = 0; i < kMatrixSize * kMatrixSize; ++i) {
    in_a[i] = static_cast<int>((kMatrixSize * kMatrixSize) - i);
  }

  // Заполняем матрицу B значениями от 1 до 4096
  for (size_t i = 0; i < kMatrixSize * kMatrixSize; ++i) {
    in_b[i] = static_cast<int>(i + 1);
  }

  // Ожидаемый результат (рассчитаем его вручную или с помощью стандартного умножения)
  std::vector<int> expected_result(kMatrixSize * kMatrixSize);

  // Рассчитываем expected_result с помощью стандартного умножения матриц
  for (size_t i = 0; i < kMatrixSize; ++i) {
    for (size_t j = 0; j < kMatrixSize; ++j) {
      int sum = 0;
      for (size_t k = 0; k < kMatrixSize; ++k) {
        sum += in_a[(i * kMatrixSize) + k] * in_b[(k * kMatrixSize) + j];
      }
      expected_result[(i * kMatrixSize) + j] = sum;
    }
  }

  // Создаем выходной вектор для результата алгоритма Штрассена
  std::vector<int> out(kMatrixSize * kMatrixSize, 0);

  // Создаем task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Создаем задачу
  nasedkin_e_strassen_algorithm_seq::StrassenSequential strassen_task_sequential(task_data_seq);
  ASSERT_EQ(strassen_task_sequential.Validation(), true);
  strassen_task_sequential.PreProcessing();
  strassen_task_sequential.Run();
  strassen_task_sequential.PostProcessing();

  // Проверяем результат
  EXPECT_EQ(expected_result, out);
}

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_64x64) {
  constexpr size_t kMatrixSize = 64;

  // Генерация случайных матриц
  std::vector<int> in_a = GenerateRandomMatrix(kMatrixSize);
  std::vector<int> in_b = GenerateRandomMatrix(kMatrixSize);
  std::vector<int> out(kMatrixSize * kMatrixSize, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  nasedkin_e_strassen_algorithm_seq::StrassenSequential strassen_task_sequential(task_data_seq);
  ASSERT_EQ(strassen_task_sequential.Validation(), true);
  strassen_task_sequential.PreProcessing();
  strassen_task_sequential.Run();
  strassen_task_sequential.PostProcessing();
}

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_128x128) {
  constexpr size_t kMatrixSize = 128;

  // Генерация случайных матриц
  std::vector<int> in_a = GenerateRandomMatrix(kMatrixSize);
  std::vector<int> in_b = GenerateRandomMatrix(kMatrixSize);
  std::vector<int> out(kMatrixSize * kMatrixSize, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  nasedkin_e_strassen_algorithm_seq::StrassenSequential strassen_task_sequential(task_data_seq);
  ASSERT_EQ(strassen_task_sequential.Validation(), true);
  strassen_task_sequential.PreProcessing();
  strassen_task_sequential.Run();
  strassen_task_sequential.PostProcessing();
}

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_256x256) {
  constexpr size_t kMatrixSize = 256;

  // Генерация случайных матриц
  std::vector<int> in_a = GenerateRandomMatrix(kMatrixSize);
  std::vector<int> in_b = GenerateRandomMatrix(kMatrixSize);
  std::vector<int> out(kMatrixSize * kMatrixSize, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  nasedkin_e_strassen_algorithm_seq::StrassenSequential strassen_task_sequential(task_data_seq);
  ASSERT_EQ(strassen_task_sequential.Validation(), true);
  strassen_task_sequential.PreProcessing();
  strassen_task_sequential.Run();
  strassen_task_sequential.PostProcessing();
}

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_512x512) {
  constexpr size_t kMatrixSize = 512;

  // Генерация случайных матриц
  std::vector<int> in_a = GenerateRandomMatrix(kMatrixSize);
  std::vector<int> in_b = GenerateRandomMatrix(kMatrixSize);
  std::vector<int> out(kMatrixSize * kMatrixSize, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  nasedkin_e_strassen_algorithm_seq::StrassenSequential strassen_task_sequential(task_data_seq);
  ASSERT_EQ(strassen_task_sequential.Validation(), true);
  strassen_task_sequential.PreProcessing();
  strassen_task_sequential.Run();
  strassen_task_sequential.PostProcessing();
}