#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/kapustin_i_jarv_alg/include/ops_tbb.hpp"

namespace {
std::vector<std::pair<int, int>> GenerateRandomPoints(size_t count, int min_x, int max_x, int min_y, int max_y) {
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> dist_x(min_x, max_x);
  std::uniform_int_distribution<int> dist_y(min_y, max_y);

  std::vector<std::pair<int, int>> random_points;
  random_points.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    random_points.emplace_back(dist_x(rng), dist_y(rng));
  }

  return random_points;
}
}  // namespace

TEST(KapustinJarvAlgTBBTest, FixedPointsWithRandomNoise) {
  std::vector<std::pair<int, int>> fixed_points = {{-1000, -1000}, {1000, -1000}, {1000, 1000}, {-1000, 1000}};

  auto random_points = GenerateRandomPoints(100, -900, 900, -900, 900);

  std::vector<std::pair<int, int>> input_points = fixed_points;
  input_points.insert(input_points.end(), random_points.begin(), random_points.end());

  std::vector<std::pair<int, int>> output_result(fixed_points.size());

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_tbb->inputs_count.emplace_back(input_points.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_tbb->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_tbb::TestTaskTBB test_task_tbb(task_data_tbb);

  ASSERT_TRUE(test_task_tbb.Validation());

  test_task_tbb.PreProcessingImpl();
  test_task_tbb.RunImpl();
  test_task_tbb.PostProcessingImpl();

  EXPECT_EQ(output_result.size(), fixed_points.size());
  for (size_t i = 0; i < fixed_points.size(); ++i) {
    EXPECT_EQ(output_result[i].first, fixed_points[i].first);
    EXPECT_EQ(output_result[i].second, fixed_points[i].second);
  }
}

TEST(KapustinJarvAlgTBBTest, HexagonWithInnerPoints) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {4, 0}, {6, 2}, {4, 4}, {0, 4}, {-2, 2},
                                                   {2, 2}, {3, 1}, {1, 3}, {3, 3}, {2, 1}};

  std::vector<std::pair<int, int>> expected_result = {{-2, 2}, {0, 0}, {4, 0}, {6, 2}, {4, 4}, {0, 4}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_tbb->inputs_count.emplace_back(input_points.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_tbb->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i].first, output_result[i].first);
    EXPECT_EQ(expected_result[i].second, output_result[i].second);
  }
}

TEST(KapustinJarvAlgTBBTest, TriangleWithInnerPoints) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {5, 8}, {10, 0}, {5, 4}, {3, 2}, {7, 2}, {5, 6}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {10, 0}, {5, 8}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_tbb->inputs_count.emplace_back(input_points.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_tbb->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgTBBTest, PureTriangle) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {5, 8}, {10, 0}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {10, 0}, {5, 8}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_tbb->inputs_count.emplace_back(input_points.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_tbb->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgTBBTest, Line) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {2, 0}, {4, 0}, {6, 0}, {8, 0}, {10, 0}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {10, 0}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_tbb->inputs_count.emplace_back(input_points.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_tbb->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgTBBTest, Rectangle) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {0, 2}, {2, 0}, {2, 2}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_tbb->inputs_count.emplace_back(input_points.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_tbb->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgTBBTest, DuplicatesAndTriangle) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {5, 8}, {10, 0}, {5, 8}, {0, 0}, {10, 0}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {10, 0}, {5, 8}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_tbb->inputs_count.emplace_back(input_points.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_tbb->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgTBBTest, DuplicatesOnly) {
  std::vector<std::pair<int, int>> input_points = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  std::vector<std::pair<int, int>> expected_result = {{1, 1}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_tbb->inputs_count.emplace_back(input_points.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_tbb->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgTBBTest, Circle) {
  std::vector<std::pair<int, int>> input_points = {{0, 5},  {3, 4},   {4, 3},   {5, 0},  {4, -3}, {3, -4},
                                                   {0, -5}, {-3, -4}, {-4, -3}, {-5, 0}, {-4, 3}, {-3, 4}};
  std::vector<std::pair<int, int>> expected_result = {{-5, 0}, {-4, -3}, {-3, -4}, {0, -5}, {3, -4}, {4, -3},
                                                      {5, 0},  {4, 3},   {3, 4},   {0, 5},  {-3, 4}, {-4, 3}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_tbb->inputs_count.emplace_back(input_points.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_tbb->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i].first, output_result[i].first);
    EXPECT_EQ(expected_result[i].second, output_result[i].second);
  }
}

TEST(KapustinJarvAlgTBBTest, Star4Points) {
  std::vector<std::pair<int, int>> input_points = {{0, 5},   {3, 2},  {5, 0},  {3, -2}, {0, -5},
                                                   {-3, -2}, {-5, 0}, {-3, 2}, {0, 0}};
  std::vector<std::pair<int, int>> expected_result = {{-5, 0}, {0, -5}, {5, 0}, {0, 5}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_tbb->inputs_count.emplace_back(input_points.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_tbb->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i].first, output_result[i].first);
    EXPECT_EQ(expected_result[i].second, output_result[i].second);
  }
}
