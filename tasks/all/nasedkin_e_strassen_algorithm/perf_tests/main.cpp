#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <random>
#include <vector>

#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"

std::vector<double> generate_matrix(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-10.0, 10.0);

  std::vector<double> matrix(size * size);
  for (int i = 0; i < size * size; ++i) {
    matrix[i] = dis(gen);
  }
  return matrix;
}

std::vector<double> standard_matrix_multiply(const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::vector<double> result(size * size, 0.0);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        result[i * size + j] += a[i * size + k] * b[k * size + j];
      }
    }
  }
  return result;
}

TEST(nasedkin_e_strassen_algorithm_all_test_pipeline_run, Test) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<int> sizes = {64, 128, 256};
    std::vector<double> times(sizes.size(), 0.0);

    for (size_t idx = 0; idx < sizes.size(); ++idx) {
      int size = sizes[idx];

      std::vector<double> matrix_a = generate_matrix(size);
      std::vector<double> matrix_b = generate_matrix(size);
      std::vector<double> out(size * size);

      auto task = std::make_shared<ppc::core::TaskData>();
      task->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_a.data()));
      task->inputs_count.emplace_back(size * size);
      task->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_b.data()));
      task->inputs_count.emplace_back(size * size);
      task->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
      task->outputs_count.emplace_back(size * size);

      auto test_task = std::make_shared<nasedkin_e_strassen_algorithm_all::StrassenAll>(task);
      ppc::core::Perf perf_analyzer(test_task);
      ppc::core::PerfAttr attr;
      attr.run_count = 5;

      ppc::core::PerfResults results;
      perf_analyzer.pipeline_run(attr, results);
      times[idx] = results.avg_time_in_ms;

      std::vector<double> expected = standard_matrix_multiply(matrix_a, matrix_b, size);
      for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(expected[i], out[i], 1e-6);
      }
    }
  }
}

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}