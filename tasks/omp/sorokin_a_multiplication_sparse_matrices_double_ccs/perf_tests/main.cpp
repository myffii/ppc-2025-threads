#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/sorokin_a_multiplication_sparse_matrices_double_ccs/include/ops_omp.hpp"

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_omp, test_pipeline_run) {
  const int msize = 20000;
  const int mrsize = 100000;

  int m = msize;
  int k = msize;
  int n = msize;

  std::vector<double> a_values(msize, 1);
  std::vector<double> a_row_indices(msize);
  for (size_t i = 0; i < msize; i++) {
    a_row_indices[i] = static_cast<int>(i);
  }
  std::vector<double> a_col_ptr(msize + 1);
  for (size_t i = 0; i <= msize; i++) {
    a_col_ptr[i] = static_cast<int>(i);
  }
  std::vector<double> b_values(msize, 1);
  std::vector<double> b_row_indices(msize);
  for (size_t i = 0; i < msize; i++) {
    b_row_indices[i] = msize - 1 - static_cast<int>(i);
  }
  std::vector<double> b_col_ptr(msize + 1);
  for (size_t i = 0; i <= msize; i++) {
    b_col_ptr[i] = static_cast<int>(i);
  }

  std::vector<double> c_values(mrsize);
  std::vector<double> c_row_indices(mrsize);
  std::vector<double> c_col_ptr(mrsize);

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(k);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_values.data()));
  task_data_omp->inputs_count.emplace_back(a_values.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_row_indices.data()));
  task_data_omp->inputs_count.emplace_back(a_row_indices.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col_ptr.data()));
  task_data_omp->inputs_count.emplace_back(a_col_ptr.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_values.data()));
  task_data_omp->inputs_count.emplace_back(b_values.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_row_indices.data()));
  task_data_omp->inputs_count.emplace_back(b_row_indices.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col_ptr.data()));
  task_data_omp->inputs_count.emplace_back(b_col_ptr.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_values.data()));
  task_data_omp->outputs_count.emplace_back(c_values.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_row_indices.data()));
  task_data_omp->outputs_count.emplace_back(c_row_indices.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_col_ptr.data()));
  task_data_omp->outputs_count.emplace_back(c_col_ptr.size());

  // Create Task
  auto test_task_omp =
      std::make_shared<sorokin_a_multiplication_sparse_matrices_double_ccs_omp::TestTaskOpenMP>(task_data_omp);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_omp, test_task_run) {
  const int msize = 20000;
  const int mrsize = 100000;

  int m = msize;
  int k = msize;
  int n = msize;

  std::vector<double> a_values(msize, 1);
  std::vector<double> a_row_indices(msize);
  for (size_t i = 0; i < msize; i++) {
    a_row_indices[i] = static_cast<int>(i);
  }
  std::vector<double> a_col_ptr(msize + 1);
  for (size_t i = 0; i <= msize; i++) {
    a_col_ptr[i] = static_cast<int>(i);
  }
  std::vector<double> b_values(msize, 1);
  std::vector<double> b_row_indices(msize);
  for (size_t i = 0; i < msize; i++) {
    b_row_indices[i] = msize - 1 - static_cast<int>(i);
  }
  std::vector<double> b_col_ptr(msize + 1);
  for (size_t i = 0; i <= msize; i++) {
    b_col_ptr[i] = static_cast<int>(i);
  }

  std::vector<double> c_values(mrsize);
  std::vector<double> c_row_indices(mrsize);
  std::vector<double> c_col_ptr(mrsize);

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(k);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_values.data()));
  task_data_omp->inputs_count.emplace_back(a_values.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_row_indices.data()));
  task_data_omp->inputs_count.emplace_back(a_row_indices.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col_ptr.data()));
  task_data_omp->inputs_count.emplace_back(a_col_ptr.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_values.data()));
  task_data_omp->inputs_count.emplace_back(b_values.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_row_indices.data()));
  task_data_omp->inputs_count.emplace_back(b_row_indices.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col_ptr.data()));
  task_data_omp->inputs_count.emplace_back(b_col_ptr.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_values.data()));
  task_data_omp->outputs_count.emplace_back(c_values.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_row_indices.data()));
  task_data_omp->outputs_count.emplace_back(c_row_indices.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_col_ptr.data()));
  task_data_omp->outputs_count.emplace_back(c_col_ptr.size());

  // Create Task
  auto test_task_omp =
      std::make_shared<sorokin_a_multiplication_sparse_matrices_double_ccs_omp::TestTaskOpenMP>(task_data_omp);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
