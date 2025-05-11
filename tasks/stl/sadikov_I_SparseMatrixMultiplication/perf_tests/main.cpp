#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/sadikov_I_SparseMatrixMultiplication/include/SparseMatrix.hpp"
#include "stl/sadikov_I_SparseMatrixMultiplication/include/ops_stl.hpp"

namespace {

enum class TestType : char { kPipline, kTaskRun };

std::vector<double> GetRandomMatrix(int size) {
  std::vector<double> data(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  int low = -5000;
  int high = 5000;
  std::uniform_int_distribution<> number(low, high);
  for (int i = 0; i < size / 5; ++i) {
    data[i] = static_cast<double>(number(gen));
  }
  std::ranges::shuffle(data, gen);
  return data;
}
struct TestData {
  std::vector<double> first_matrix;
  int first_matrix_rows_count = 0;
  int first_matrix_columns_count = 0;
  std::vector<double> second_matrix;
  int second_matrix_rows_count = 0;
  int second_matrix_columns_count = 0;
  std::vector<double> multiplication_result;
};

// NOLINTNEXTLINE(readability-identifier-naming)
class sadikov_i_matrix_multiplication_testing_stl : public testing::Test {
  static constexpr auto kEpsilon = 0.000001;

 protected:
  std::shared_ptr<ppc::core::TaskData> m_task_data_stl;

 public:
  // NOLINTNEXTLINE(readability-identifier-naming)
  sadikov_i_matrix_multiplication_testing_stl() { m_task_data_stl = std::make_shared<ppc::core::TaskData>(); }

  void FillTaskData(TestData& data) {
    m_task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.first_matrix.data()));
    m_task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.second_matrix.data()));
    m_task_data_stl->inputs_count.emplace_back(data.first_matrix_rows_count);
    m_task_data_stl->inputs_count.emplace_back(data.first_matrix_columns_count);
    m_task_data_stl->inputs_count.emplace_back(data.second_matrix_rows_count);
    m_task_data_stl->inputs_count.emplace_back(data.second_matrix_columns_count);
    m_task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(data.multiplication_result.data()));
    m_task_data_stl->outputs_count.emplace_back(data.first_matrix_rows_count * data.second_matrix_columns_count);
  }

  void RunTask(const std::vector<double>& result_checker, const TestData& data, TestType type) {
    auto test_task_stl =
        std::make_shared<sadikov_i_sparse_matrix_multiplication_task_stl::CCSMatrixSTL>(m_task_data_stl);
    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };
    auto perf_results = std::make_shared<ppc::core::PerfResults>();
    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_stl);
    switch (type) {
      case TestType::kPipline:
        perf_analyzer->PipelineRun(perf_attr, perf_results);
        break;
      case TestType::kTaskRun:
        perf_analyzer->TaskRun(perf_attr, perf_results);
        break;
    }
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    for (size_t i = 0; i < result_checker.size(); ++i) {
      EXPECT_NEAR(data.multiplication_result[i], result_checker[i], kEpsilon);
    }
  }
};
}  // namespace

TEST_F(sadikov_i_matrix_multiplication_testing_stl, test_pipline_run) {
  constexpr int kSize = 300;
  TestData test_data = {.first_matrix = GetRandomMatrix(kSize * kSize),
                        .first_matrix_rows_count = kSize,
                        .first_matrix_columns_count = kSize,
                        .second_matrix = GetRandomMatrix(kSize * kSize),
                        .second_matrix_rows_count = kSize,
                        .second_matrix_columns_count = kSize,
                        .multiplication_result = std::vector<double>(kSize * kSize)};
  std::vector<double> test_out = sadikov_i_sparse_matrix_multiplication_task_stl::BaseMatrixMultiplication(
      test_data.first_matrix, kSize, kSize, test_data.second_matrix, kSize, kSize);
  FillTaskData(test_data);
  RunTask(test_out, test_data, TestType::kPipline);
}

TEST_F(sadikov_i_matrix_multiplication_testing_stl, test_task_run) {
  constexpr int kSize = 300;
  TestData test_data = {.first_matrix = GetRandomMatrix(kSize * kSize),
                        .first_matrix_rows_count = kSize,
                        .first_matrix_columns_count = kSize,
                        .second_matrix = GetRandomMatrix(kSize * kSize),
                        .second_matrix_rows_count = kSize,
                        .second_matrix_columns_count = kSize,
                        .multiplication_result = std::vector<double>(kSize * kSize)};
  std::vector<double> test_out = sadikov_i_sparse_matrix_multiplication_task_stl::BaseMatrixMultiplication(
      test_data.first_matrix, kSize, kSize, test_data.second_matrix, kSize, kSize);
  FillTaskData(test_data);
  RunTask(test_out, test_data, TestType::kTaskRun);
}
