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

std::vector<int> MultiplyMatricesStandard(const std::vector<int>& a, const std::vector<int>& b, size_t size) {
  std::vector<int> result(size * size, 0);
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      for (size_t k = 0; k < size; ++k) {
        result[(i * size) + j] += a[(i * size) + k] * b[(k * size) + j];
      }
    }
  }
  return result;
}
}  // namespace

TEST(nasedkin_e_strassen_algorithm_seq, test_matrix_4x4_fixed) {
  constexpr size_t kMatrixSize = 5;

  // Задаем фиксированные матрицы A и B размером 5x5
  std::vector<int> in_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  std::vector<int> in_b = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  // Вычисляем ожидаемый результат с помощью стандартного умножения матриц
  std::vector<int> expected_result = MultiplyMatricesStandard(in_a, in_b, kMatrixSize);

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

TEST(nasedkin_e_strassen_algorithm_seq, test_matrix_50x50) {
  constexpr size_t kMatrixSize = 50;

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

  // Проверка результата (опционально, если есть ожидаемый результат)
  // EXPECT_EQ(expected_result, out);
}