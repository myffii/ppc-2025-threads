#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <random>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
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

TEST(nasedkin_e_strassen_algorithm_seq, test_pipeline_run) {
    constexpr int kCount = 128; // Размер матрицы (kCount x kCount)

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
    auto test_task_sequential = std::make_shared<nasedkin_e_strassen_algorithm_seq::StrassenSequential>(task_data_seq);

    // Create Perf attributes
    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
        auto current_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
        return static_cast<double>(duration) * 1e-9;
    };

    // Create and init perf results
    auto perf_results = std::make_shared<ppc::core::PerfResults>();

    // Create Perf analyzer
    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
    perf_analyzer->PipelineRun(perf_attr, perf_results);
    ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(nasedkin_e_strassen_algorithm_seq, test_task_run) {
    constexpr int kCount = 128; // Размер матрицы (kCount x kCount)

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
    auto test_task_sequential = std::make_shared<nasedkin_e_strassen_algorithm_seq::StrassenSequential>(task_data_seq);

    // Create Perf attributes
    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
        auto current_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
        return static_cast<double>(duration) * 1e-9;
    };

    // Create and init perf results
    auto perf_results = std::make_shared<ppc::core::PerfResults>();

    // Create Perf analyzer
    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
    perf_analyzer->TaskRun(perf_attr, perf_results);
    ppc::core::Perf::PrintPerfStatistic(perf_results);
}