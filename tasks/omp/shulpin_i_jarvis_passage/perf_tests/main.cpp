#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/shulpin_i_jarvis_passage/include/ops_omp.hpp"

namespace {
std::vector<shulpin_i_jarvis_omp::Point> GenerateRandomPoints(size_t num_points) {
  std::vector<shulpin_i_jarvis_omp::Point> points;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-10000, 10000);

  for (size_t i = 0; i < num_points; ++i) {
    double x = dist(gen);
    double y = dist(gen);
    points.emplace_back(x, y);
  }

  return points;
}

bool ValidateConvexHull(const std::vector<shulpin_i_jarvis_omp::Point> &hull, const size_t size) {
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

TEST(shulpin_i_jarvis_omp, test_pipeline_run) {
  size_t num_points = 1000000;
  std::vector<shulpin_i_jarvis_omp::Point> input = GenerateRandomPoints(num_points);
  std::vector<shulpin_i_jarvis_omp::Point> out(input.size());

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_par->inputs_count.emplace_back(input.size());
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_par->outputs_count.emplace_back(out.size());

  auto test_task_omp = std::make_shared<shulpin_i_jarvis_omp::JarvisOMPParallel>(task_data_par);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_TRUE(ValidateConvexHull(out, task_data_par->outputs_count[0]));
}

TEST(shulpin_i_jarvis_omp, test_task_run) {
  size_t num_points = 1000000;
  std::vector<shulpin_i_jarvis_omp::Point> input = GenerateRandomPoints(num_points);

  std::vector<shulpin_i_jarvis_omp::Point> out(input.size());

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_par->inputs_count.emplace_back(input.size());
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_par->outputs_count.emplace_back(out.size());

  auto test_task_omp = std::make_shared<shulpin_i_jarvis_omp::JarvisOMPParallel>(task_data_par);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_TRUE(ValidateConvexHull(out, task_data_par->outputs_count[0]));
}
