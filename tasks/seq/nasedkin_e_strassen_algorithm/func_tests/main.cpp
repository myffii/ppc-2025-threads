#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <random>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"

// Метод для генерации случайной матрицы
std::vector<int> generateRandomMatrix(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 100);

    std::vector<int> matrix(size * size);
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = distrib(gen);
    }
    return matrix;
}

TEST(nasedkin_e_strassen_algorithm_seq, test_matmul_50) {
    constexpr size_t kCount = 50;

    // Генерация случайных матриц
    std::vector<int> in_A = generateRandomMatrix(kCount);
    std::vector<int> in_B = generateRandomMatrix(kCount);
    std::vector<int> out(kCount * kCount, 0);

    // Create task_data
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_A.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_B.data()));
    task_data_seq->inputs_count.emplace_back(in_A.size());
    task_data_seq->inputs_count.emplace_back(in_B.size());
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

TEST(nasedkin_e_strassen_algorithm_seq, test_matmul_100_from_file) {
    std::string line;
    std::ifstream test_file(ppc::util::GetAbsolutePath("seq/nasedkin_e_strassen_algorithm/data/test.txt"));
    if (test_file.is_open()) {
        getline(test_file, line);
    }
    test_file.close();

    const size_t count = std::stoi(line);

    // Генерация случайных матриц
    std::vector<int> in_A = generateRandomMatrix(count);
    std::vector<int> in_B = generateRandomMatrix(count);
    std::vector<int> out(count * count, 0);

    // Create task_data
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_A.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_B.data()));
    task_data_seq->inputs_count.emplace_back(in_A.size());
    task_data_seq->inputs_count.emplace_back(in_B.size());
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