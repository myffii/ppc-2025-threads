#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <random>
#include <vector>

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

std::vector<shulpin_i_jarvis_omp::Point> GeneratePointsInCircle(size_t num_points,
                                                                const shulpin_i_jarvis_omp::Point &center,
                                                                double radius) {
  std::vector<shulpin_i_jarvis_omp::Point> points;
  for (size_t i = 0; i < num_points; ++i) {
    double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(num_points);
    double x = center.x + (radius * std::cos(angle));
    double y = center.y + (radius * std::sin(angle));
    points.emplace_back(x, y);
  }
  return points;
}

void VerifyResults(const std::vector<shulpin_i_jarvis_omp::Point> &expected,
                   const std::vector<shulpin_i_jarvis_omp::Point> &result_seq,
                   const std::vector<shulpin_i_jarvis_omp::Point> &result_omp) {
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(expected[i].x, result_seq[i].x);
    ASSERT_EQ(expected[i].y, result_seq[i].y);
    ASSERT_EQ(expected[i].x, result_omp[i].x);
    ASSERT_EQ(expected[i].y, result_omp[i].y);
  }
}

void VerifyResultsRandom(const std::vector<shulpin_i_jarvis_omp::Point> &expected,
                         const std::vector<shulpin_i_jarvis_omp::Point> &result_omp) {
  for (const auto &p : result_omp) {
    bool found = false;
    for (const auto &q : expected) {
      if (std::fabs(p.x - q.x) < 1e-6 && std::fabs(p.y - q.y) < 1e-6) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found);
  }
}

inline size_t CalculateIndex(size_t i, size_t tmp) { return (i < tmp) ? (i + tmp) : (i - tmp); }

inline void ExpectEqualPoints(const shulpin_i_jarvis_omp::Point &expected, const shulpin_i_jarvis_omp::Point &seq,
                              const shulpin_i_jarvis_omp::Point &omp) {
  EXPECT_EQ(expected.x, seq.x);
  EXPECT_EQ(expected.y, seq.y);
  EXPECT_EQ(expected.x, omp.x);
  EXPECT_EQ(expected.y, omp.y);
}

void VerifyResultsCircle(const std::vector<shulpin_i_jarvis_omp::Point> &expected,
                         const std::vector<shulpin_i_jarvis_omp::Point> &result_seq,
                         const std::vector<shulpin_i_jarvis_omp::Point> &result_omp, size_t &num_points) {
  size_t tmp = num_points >> 1;

  for (size_t i = 0; i < expected.size(); ++i) {
    size_t idx = CalculateIndex(i, tmp);
    ExpectEqualPoints(expected[i], result_seq[idx], result_omp[idx]);
  }
}

int Orientation(const shulpin_i_jarvis_omp::Point &p, const shulpin_i_jarvis_omp::Point &q,
                const shulpin_i_jarvis_omp::Point &r) {
  double val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y));
  if (std::fabs(val) < 1e-9) {
    return 0;
  }
  return (val > 0) ? 1 : 2;
}

std::vector<shulpin_i_jarvis_omp::Point> ComputeConvexHull(std::vector<shulpin_i_jarvis_omp::Point> raw_points) {
  std::vector<shulpin_i_jarvis_omp::Point> convex_shell{};
  const size_t count = raw_points.size();

  size_t ref_idx = 0;
  for (size_t idx = 1; idx < count; ++idx) {
    const auto &p = raw_points[idx];
    const auto &ref = raw_points[ref_idx];
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

void MainTestBody(std::vector<shulpin_i_jarvis_omp::Point> &input, std::vector<shulpin_i_jarvis_omp::Point> &expected) {
  std::vector<shulpin_i_jarvis_omp::Point> result_seq(expected.size());
  std::vector<shulpin_i_jarvis_omp::Point> result_omp(expected.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_seq.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<uint32_t>(result_seq.size()));

  shulpin_i_jarvis_omp::JarvisSequential seq_task(task_data_seq);
  ASSERT_EQ(seq_task.Validation(), true);
  seq_task.PreProcessing();
  seq_task.Run();
  seq_task.PostProcessing();

  auto task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_par->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_omp.data()));
  task_data_par->outputs_count.emplace_back(static_cast<uint32_t>(result_omp.size()));

  shulpin_i_jarvis_omp::JarvisOMPParallel omp_task(task_data_par);
  ASSERT_EQ(omp_task.Validation(), true);
  omp_task.PreProcessing();
  omp_task.Run();
  omp_task.PostProcessing();

  VerifyResults(expected, result_seq, result_omp);
}

void TestBodyFalse(std::vector<shulpin_i_jarvis_omp::Point> &input,
                   std::vector<shulpin_i_jarvis_omp::Point> &expected) {
  std::vector<shulpin_i_jarvis_omp::Point> result_omp(expected.size());

  auto task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_par->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_omp.data()));
  task_data_par->outputs_count.emplace_back(static_cast<uint32_t>(result_omp.size()));

  shulpin_i_jarvis_omp::JarvisOMPParallel omp_task(task_data_par);
  ASSERT_EQ(omp_task.Validation(), false);
}

void TestBodyRandomCircle(std::vector<shulpin_i_jarvis_omp::Point> &input,
                          std::vector<shulpin_i_jarvis_omp::Point> &expected, size_t &num_points) {
  std::vector<shulpin_i_jarvis_omp::Point> result_seq(expected.size());
  std::vector<shulpin_i_jarvis_omp::Point> result_omp(expected.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_seq.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<uint32_t>(result_seq.size()));

  shulpin_i_jarvis_omp::JarvisSequential seq_task(task_data_seq);
  ASSERT_EQ(seq_task.Validation(), true);
  seq_task.PreProcessing();
  seq_task.Run();
  seq_task.PostProcessing();

  auto task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_par->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_omp.data()));
  task_data_par->outputs_count.emplace_back(static_cast<uint32_t>(result_omp.size()));

  shulpin_i_jarvis_omp::JarvisOMPParallel omp_task(task_data_par);
  ASSERT_EQ(omp_task.Validation(), true);
  omp_task.PreProcessing();
  omp_task.Run();
  omp_task.PostProcessing();

  VerifyResultsCircle(expected, result_seq, result_omp, num_points);
}

void RandomTestBody(std::vector<shulpin_i_jarvis_omp::Point> &input,
                    std::vector<shulpin_i_jarvis_omp::Point> &expected) {
  std::vector<shulpin_i_jarvis_omp::Point> result_omp(expected.size());

  auto task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_par->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_omp.data()));
  task_data_par->outputs_count.emplace_back(static_cast<uint32_t>(result_omp.size()));

  shulpin_i_jarvis_omp::JarvisOMPParallel omp_task(task_data_par);
  ASSERT_EQ(omp_task.Validation(), true);
  omp_task.PreProcessing();
  omp_task.Run();
  omp_task.PostProcessing();

  VerifyResultsRandom(expected, result_omp);
}
}  // namespace

TEST(shulpin_i_jarvis_omp, square_with_point) {
  std::vector<shulpin_i_jarvis_omp::Point> input = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {1, 1}};
  std::vector<shulpin_i_jarvis_omp::Point> expected = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_omp, ox_line) {
  std::vector<shulpin_i_jarvis_omp::Point> input = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};
  std::vector<shulpin_i_jarvis_omp::Point> expected = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_omp, triangle) {
  std::vector<shulpin_i_jarvis_omp::Point> input = {{0, 0}, {3, 0}, {1, 2}};
  std::vector<shulpin_i_jarvis_omp::Point> expected = {{0, 0}, {3, 0}, {1, 2}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_omp, octagone) {
  std::vector<shulpin_i_jarvis_omp::Point> input = {{1, 0}, {2, 0}, {3, 1}, {3, 2}, {2, 3}, {1, 3}, {0, 2}, {0, 1}};
  std::vector<shulpin_i_jarvis_omp::Point> expected = {{0, 1}, {1, 0}, {2, 0}, {3, 1}, {3, 2}, {2, 3}, {1, 3}, {0, 2}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_omp, repeated_points) {
  std::vector<shulpin_i_jarvis_omp::Point> input = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {2, 0}, {0, 0}};
  std::vector<shulpin_i_jarvis_omp::Point> expected = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_omp, real_case) {
  std::vector<shulpin_i_jarvis_omp::Point> input = {{1, 1}, {3, 2}, {5, 1}, {4, 3}, {2, 4}, {1, 3}, {3, 3}};
  std::vector<shulpin_i_jarvis_omp::Point> expected = {{1, 1}, {5, 1}, {4, 3}, {2, 4}, {1, 3}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_omp, star_case) {
  // clang-format off
  std::vector<shulpin_i_jarvis_omp::Point> input = {{0.0, 3.0},
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
  std::vector<shulpin_i_jarvis_omp::Point> expected = {{-3.0, 1.0},
      {-2.5, -3.0},
      {2.5, -3.0},
      {3.0, 1.0},
      {0.0, 3.0},
  };
  // clang-format on
  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_omp, one_point_validation_false) {
  std::vector<shulpin_i_jarvis_omp::Point> input = {{0, 0}};
  std::vector<shulpin_i_jarvis_omp::Point> expected = {{0, 0}};

  TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_omp, three_points_validation_false) {
  std::vector<shulpin_i_jarvis_omp::Point> input = {{1, 1}, {2, 2}};
  std::vector<shulpin_i_jarvis_omp::Point> expected = {{1, 1}, {2, 2}};

  TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_omp, zero_points_validation_false) {
  std::vector<shulpin_i_jarvis_omp::Point> input = {};
  std::vector<shulpin_i_jarvis_omp::Point> expected = {};

  TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_omp, circle_r10_p100) {
  shulpin_i_jarvis_omp::Point center{0, 0};

  double radius = 10.0;
  size_t num_points = 100;

  std::vector<shulpin_i_jarvis_omp::Point> input = GeneratePointsInCircle(num_points, center, radius);
  std::vector<shulpin_i_jarvis_omp::Point> expected = input;

  TestBodyRandomCircle(input, expected, num_points);
}

TEST(shulpin_i_jarvis_omp, circle_r10_p200) {
  shulpin_i_jarvis_omp::Point center{0, 0};

  double radius = 10.0;
  size_t num_points = 200;

  std::vector<shulpin_i_jarvis_omp::Point> input = GeneratePointsInCircle(num_points, center, radius);
  std::vector<shulpin_i_jarvis_omp::Point> expected = input;

  TestBodyRandomCircle(input, expected, num_points);
}

TEST(shulpin_i_jarvis_omp, random_5_points) {
  size_t num_points = 5;

  std::vector<shulpin_i_jarvis_omp::Point> input = GenerateRandomPoints(num_points);
  std::vector<shulpin_i_jarvis_omp::Point> expected = ComputeConvexHull(input);

  RandomTestBody(input, expected);
}

TEST(shulpin_i_jarvis_omp, random_10_points) {
  size_t num_points = 10;

  std::vector<shulpin_i_jarvis_omp::Point> input = GenerateRandomPoints(num_points);
  std::vector<shulpin_i_jarvis_omp::Point> expected = ComputeConvexHull(input);

  RandomTestBody(input, expected);
}

TEST(shulpin_i_jarvis_omp, random_50_points) {
  size_t num_points = 50;

  std::vector<shulpin_i_jarvis_omp::Point> input = GenerateRandomPoints(num_points);
  std::vector<shulpin_i_jarvis_omp::Point> expected = ComputeConvexHull(input);

  RandomTestBody(input, expected);
}

TEST(shulpin_i_jarvis_omp, random_100_points) {
  size_t num_points = 100;

  std::vector<shulpin_i_jarvis_omp::Point> input = GenerateRandomPoints(num_points);
  std::vector<shulpin_i_jarvis_omp::Point> expected = ComputeConvexHull(input);

  RandomTestBody(input, expected);
}
