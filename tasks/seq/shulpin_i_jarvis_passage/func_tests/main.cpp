#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/shulpin_i_jarvis_passage/include/ops_seq.hpp"

namespace {
std::vector<shulpin_i_jarvis_seq::Point> GeneratePointsInCircle(size_t num_points,
                                                                const shulpin_i_jarvis_seq::Point& center,
                                                                double radius) {
  std::vector<shulpin_i_jarvis_seq::Point> points;
  for (size_t i = 0; i < num_points; ++i) {
    double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(num_points);
    double x = center.x + (radius * std::cos(angle));
    double y = center.y + (radius * std::sin(angle));
    points.emplace_back(x, y);
  }
  return points;
}

std::vector<shulpin_i_jarvis_seq::Point> GenerateRandomPoints(size_t num_points) {
  std::vector<shulpin_i_jarvis_seq::Point> points;
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

int Orientation(const shulpin_i_jarvis_seq::Point& p, const shulpin_i_jarvis_seq::Point& q,
                const shulpin_i_jarvis_seq::Point& r) {
  double val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y));
  if (std::fabs(val) < 1e-9) {
    return 0;
  }
  return (val > 0) ? 1 : 2;
}

std::vector<shulpin_i_jarvis_seq::Point> ComputeConvexHull(std::vector<shulpin_i_jarvis_seq::Point> raw_points) {
  std::vector<shulpin_i_jarvis_seq::Point> convex_shell{};
  const size_t count = raw_points.size();

  size_t ref_idx = 0;
  for (size_t idx = 1; idx < count; ++idx) {
    const auto& p = raw_points[idx];
    const auto& ref = raw_points[ref_idx];
    if ((p.x < ref.x) || (p.x == ref.x && p.y < ref.y)) {
      ref_idx = idx;
    }
  }

  std::vector<bool> included(count, false);
  size_t current = ref_idx;

  while (true) {
    convex_shell.push_back(raw_points[current]);
    included[current] = true;

    size_t next = (current + 1) % count;

    for (size_t trial = 0; trial < count; ++trial) {
      if (trial == current || trial == next) {
        continue;
      }

      int orient = Orientation(raw_points[current], raw_points[trial], raw_points[next]);
      if (orient == 2) {
        next = trial;
      }
    }

    current = next;
    if (current == ref_idx) {
      break;
    }
  }
  return convex_shell;
}

void MainTestBody(std::vector<shulpin_i_jarvis_seq::Point>& input, std::vector<shulpin_i_jarvis_seq::Point>& expected) {
  std::vector<shulpin_i_jarvis_seq::Point> result(expected.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_seq->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<uint32_t>(result.size()));

  shulpin_i_jarvis_seq::JarvisSequential seq_task(task_data_seq);
  ASSERT_EQ(seq_task.Validation(), true);
  seq_task.PreProcessing();
  seq_task.Run();
  seq_task.PostProcessing();

  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(expected[i].x, result[i].x);
    ASSERT_EQ(expected[i].y, result[i].y);
  }
}

void TestBodyFalse(std::vector<shulpin_i_jarvis_seq::Point>& input,
                   std::vector<shulpin_i_jarvis_seq::Point>& expected) {
  std::vector<shulpin_i_jarvis_seq::Point> result(expected.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_seq->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<uint32_t>(result.size()));

  shulpin_i_jarvis_seq::JarvisSequential seq_task(task_data_seq);
  ASSERT_EQ(seq_task.Validation(), false);
}

void TestBodyRandomCircle(std::vector<shulpin_i_jarvis_seq::Point>& input,
                          std::vector<shulpin_i_jarvis_seq::Point>& expected, size_t& num_points) {
  std::vector<shulpin_i_jarvis_seq::Point> result(expected.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_seq->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<uint32_t>(result.size()));

  shulpin_i_jarvis_seq::JarvisSequential seq_task(task_data_seq);
  ASSERT_EQ(seq_task.Validation(), true);
  seq_task.PreProcessing();
  seq_task.Run();
  seq_task.PostProcessing();

  size_t tmp = num_points >> 1;

  for (size_t i = 0; i < result.size(); ++i) {
    size_t idx = (i < tmp) ? (i + tmp) : (i - tmp);
    EXPECT_EQ(expected[i].x, result[idx].x);
    EXPECT_EQ(expected[i].y, result[idx].y);
  }
}

void RandomTestBody(std::vector<shulpin_i_jarvis_seq::Point>& input,
                    std::vector<shulpin_i_jarvis_seq::Point>& expected) {
  std::vector<shulpin_i_jarvis_seq::Point> result(expected.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_seq->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<uint32_t>(result.size()));

  shulpin_i_jarvis_seq::JarvisSequential seq_task(task_data_seq);
  ASSERT_EQ(seq_task.Validation(), true);
  seq_task.PreProcessing();
  seq_task.Run();
  seq_task.PostProcessing();

  for (const auto& p : result) {
    bool found = false;
    for (const auto& q : expected) {
      if (std::fabs(p.x - q.x) < 1e-6 && std::fabs(p.y - q.y) < 1e-6) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found);
  }
}
}  // namespace

TEST(shulpin_i_jarvis_seq, square_with_point) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {1, 1}};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_seq, ox_line) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_seq, triangle) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {{0, 0}, {3, 0}, {1, 2}};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{0, 0}, {3, 0}, {1, 2}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_seq, octagone) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {{1, 0}, {2, 0}, {3, 1}, {3, 2}, {2, 3}, {1, 3}, {0, 2}, {0, 1}};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{0, 1}, {1, 0}, {2, 0}, {3, 1}, {3, 2}, {2, 3}, {1, 3}, {0, 2}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_seq, repeated_points) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {2, 0}, {0, 0}};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_seq, real_case) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {{1, 1}, {3, 2}, {5, 1}, {4, 3}, {2, 4}, {1, 3}, {3, 3}};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{1, 1}, {5, 1}, {4, 3}, {2, 4}, {1, 3}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_seq, star_case) {
  // clang-format off
  std::vector<shulpin_i_jarvis_seq::Point> input = {{0.0, 3.0},
    {1.0, 1.0},
    {3.0, 1.0},
    {1.5, -0.5},
    {2.5, -3.0},
    {0.0, -1.5},
    {-2.5, -3.0},
    {-1.5, -0.5},
    {-3.0, 1.0},
    {-1.0, 1.0},
    {0.0, 3.0}
  };
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{-3.0, 1.0},
      {-2.5, -3.0},
      {2.5, -3.0},
      {3.0, 1.0},
      {0.0, 3.0},
  };
  // clang-format on
  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_seq, one_point_validation_false) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {{0, 0}};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{0, 0}};

  TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_seq, three_points_validation_false) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {{1, 1}, {2, 2}};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{1, 1}, {2, 2}};

  TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_seq, zero_points_validation_false) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {};

  TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_seq, circle_r10_p100) {
  shulpin_i_jarvis_seq::Point center{0, 0};

  double radius = 10.0;
  size_t num_points = 100;

  std::vector<shulpin_i_jarvis_seq::Point> input = GeneratePointsInCircle(num_points, center, radius);
  std::vector<shulpin_i_jarvis_seq::Point> expected = input;

  TestBodyRandomCircle(input, expected, num_points);
}

TEST(shulpin_i_jarvis_seq, circle_r10_p1000) {
  shulpin_i_jarvis_seq::Point center{0, 0};

  double radius = 10.0;
  size_t num_points = 1000;

  std::vector<shulpin_i_jarvis_seq::Point> input = GeneratePointsInCircle(num_points, center, radius);
  std::vector<shulpin_i_jarvis_seq::Point> expected = input;

  TestBodyRandomCircle(input, expected, num_points);
}

TEST(shulpin_i_jarvis_seq, random_5_points) {
  size_t num_points = 5;

  std::vector<shulpin_i_jarvis_seq::Point> input = GenerateRandomPoints(num_points);
  std::vector<shulpin_i_jarvis_seq::Point> expected = ComputeConvexHull(input);

  RandomTestBody(input, expected);
}

TEST(shulpin_i_jarvis_seq, random_10_points) {
  size_t num_points = 10;

  std::vector<shulpin_i_jarvis_seq::Point> input = GenerateRandomPoints(num_points);
  std::vector<shulpin_i_jarvis_seq::Point> expected = ComputeConvexHull(input);

  RandomTestBody(input, expected);
}

TEST(shulpin_i_jarvis_seq, random_50_points) {
  size_t num_points = 50;

  std::vector<shulpin_i_jarvis_seq::Point> input = GenerateRandomPoints(num_points);
  std::vector<shulpin_i_jarvis_seq::Point> expected = ComputeConvexHull(input);

  RandomTestBody(input, expected);
}

TEST(shulpin_i_jarvis_seq, random_100_points) {
  size_t num_points = 100;

  std::vector<shulpin_i_jarvis_seq::Point> input = GenerateRandomPoints(num_points);
  std::vector<shulpin_i_jarvis_seq::Point> expected = ComputeConvexHull(input);

  RandomTestBody(input, expected);
}
