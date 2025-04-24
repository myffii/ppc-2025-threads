#include <gtest/gtest.h>
#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/zinoviev_a_convex_hull_components/include/ops_omp.hpp"

namespace {
void CheckResult(const std::vector<zinoviev_a_convex_hull_components_omp::Point>& result,
                 const std::vector<zinoviev_a_convex_hull_components_omp::Point>& expect) {
  ASSERT_EQ(result.size(), expect.size());
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i].x, expect[i].x);
    ASSERT_EQ(result[i].y, expect[i].y);
  }
}
}  // namespace

TEST(zinoviev_a_convex_hull_omp, test_pipeline_run) {
  const int size = 5000;
  std::vector<int> input(size * size, 1);

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  data->inputs_count.push_back(size);
  data->inputs_count.push_back(size);
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_omp::Point[4]));
  data->outputs_count.push_back(4);

  auto task = std::make_shared<zinoviev_a_convex_hull_components_omp::ConvexHullOMP>(data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 3;
  perf_attr->current_timer = [] { return omp_get_wtime(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  auto* res = reinterpret_cast<zinoviev_a_convex_hull_components_omp::Point*>(data->outputs[0]);
  size_t res_size = data->outputs_count[0];
  std::vector<zinoviev_a_convex_hull_components_omp::Point> actual(res, res + res_size);

  std::vector<zinoviev_a_convex_hull_components_omp::Point> expect{
      {.x = 0, .y = 0}, {.x = 4999, .y = 0}, {.x = 4998, .y = 4999}, {.x = 0, .y = 4999}};
  CheckResult(actual, expect);

  delete[] reinterpret_cast<zinoviev_a_convex_hull_components_omp::Point*>(data->outputs[0]);
}

TEST(zinoviev_a_convex_hull_omp, test_task_run) {
  const int size = 5000;
  std::vector<int> input(size * size, 1);

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  data->inputs_count.push_back(size);
  data->inputs_count.push_back(size);
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_omp::Point[4]));
  data->outputs_count.push_back(4);

  auto task = std::make_shared<zinoviev_a_convex_hull_components_omp::ConvexHullOMP>(data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 3;
  perf_attr->current_timer = [] { return omp_get_wtime(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  auto* res = reinterpret_cast<zinoviev_a_convex_hull_components_omp::Point*>(data->outputs[0]);
  size_t res_size = data->outputs_count[0];
  std::vector<zinoviev_a_convex_hull_components_omp::Point> actual(res, res + res_size);

  std::vector<zinoviev_a_convex_hull_components_omp::Point> expect{
      {.x = 0, .y = 0}, {.x = 4999, .y = 0}, {.x = 4998, .y = 4999}, {.x = 0, .y = 4999}};
  CheckResult(actual, expect);

  delete[] reinterpret_cast<zinoviev_a_convex_hull_components_omp::Point*>(data->outputs[0]);
}