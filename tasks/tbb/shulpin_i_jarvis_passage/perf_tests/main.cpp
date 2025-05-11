#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/shulpin_i_jarvis_passage/include/ops_tbb.hpp"
#include "tbb/shulpin_i_jarvis_passage/include/test_modules.hpp"

constexpr int k_ = 100;
constexpr int kBound = 1000;

TEST(shulpin_i_jarvis_tbb, test_pipeline_run) {
  size_t num_points = 5000000;
  std::vector<shulpin_i_jarvis_tbb::Point> hull = {
      {-kBound, -kBound}, {kBound, -kBound}, {kBound, kBound}, {-kBound, kBound}};
  std::vector<shulpin_i_jarvis_tbb::Point> input =
      shulpin_tbb_test_module::PerfRandomGenerator(num_points, -kBound + k_, kBound - k_);
  std::vector<shulpin_i_jarvis_tbb::Point> out(input.size());
  input.insert(input.end(), hull.begin(), hull.end());

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_par->inputs_count.emplace_back(input.size());
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_par->outputs_count.emplace_back(out.size());

  auto test_task_tbb = std::make_shared<shulpin_i_jarvis_tbb::JarvisTBBParallel>(task_data_par);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (size_t i = 0; i < hull.size(); ++i) {
    EXPECT_EQ(hull[i].x, out[i].x);
    EXPECT_EQ(hull[i].y, out[i].y);
  }
}

TEST(shulpin_i_jarvis_tbb, test_task_run) {
  size_t num_points = 5000000;
  std::vector<shulpin_i_jarvis_tbb::Point> hull = {
      {-kBound, -kBound}, {kBound, -kBound}, {kBound, kBound}, {-kBound, kBound}};
  std::vector<shulpin_i_jarvis_tbb::Point> input =
      shulpin_tbb_test_module::PerfRandomGenerator(num_points, -kBound + k_, kBound - k_);
  std::vector<shulpin_i_jarvis_tbb::Point> out(input.size());
  input.insert(input.end(), hull.begin(), hull.end());

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_par->inputs_count.emplace_back(input.size());
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_par->outputs_count.emplace_back(out.size());

  auto test_task_tbb = std::make_shared<shulpin_i_jarvis_tbb::JarvisTBBParallel>(task_data_par);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (size_t i = 0; i < hull.size(); ++i) {
    EXPECT_EQ(hull[i].x, out[i].x);
    EXPECT_EQ(hull[i].y, out[i].y);
  }
}
