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

TEST(nasedkin_e_strassen_algorithm_seq, test_matmul_50) {
  constexpr size_t kCount = 50;

  // Генерация случайных матриц
  std::vector<int> in_a = GenerateRandomMatrix(kCount);
  std::vector<int> in_b = GenerateRandomMatrix(kCount);
  std::vector<int> out(kCount * kCount, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  nasedkin_e_strassen_algorithm_seq::StrassenSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  // Проверка результата (опционально, если есть ожидаемый результат)
  // EXPECT_EQ(expected_result, out);
}