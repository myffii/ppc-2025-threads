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

namespace {
bool ValidateConvexHull(const std::vector<shulpin_i_jarvis_tbb::Point> &hull, const size_t size) {
  if (hull.size() < 3) {
    return false;
  }

  for (size_t i = 0; i < size; ++i) {
    const auto &p1 = hull[i];
    const auto &p2 = hull[(i + 1) % size];
    const auto &p3 = hull[(i + 2) % size];

    double cross = ((p2.x - p1.x) * (p3.y - p1.y)) - ((p3.x - p1.x) * (p2.y - p1.y));
    if (cross < 0.0) {
      return false;
    }
  }

  return true;
}
}  // namespace

TEST(shulpin_i_jarvis_tbb, test_pipeline_run) {
  size_t num_points = 1000000;
  std::vector<shulpin_i_jarvis_tbb::Point> input = shulpin_tbb_test_module::GenerateRandomPoints(num_points);
  std::vector<shulpin_i_jarvis_tbb::Point> out(input.size());

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

  ASSERT_TRUE(ValidateConvexHull(out, task_data_par->outputs_count[0]));
}

TEST(shulpin_i_jarvis_tbb, test_task_run) {
  size_t num_points = 1000000;
  std::vector<shulpin_i_jarvis_tbb::Point> input = shulpin_tbb_test_module::GenerateRandomPoints(num_points);

  std::vector<shulpin_i_jarvis_tbb::Point> out(input.size());

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

  ASSERT_TRUE(ValidateConvexHull(out, task_data_par->outputs_count[0]));
}
