#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "tbb/shulpin_i_jarvis_passage/include/ops_tbb.hpp"
#include "tbb/shulpin_i_jarvis_passage/include/test_modules.hpp"

TEST(shulpin_i_jarvis_tbb, square_with_point) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {1, 1}};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};

  shulpin_tbb_test_module::MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_tbb, ox_line) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};

  shulpin_tbb_test_module::MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_tbb, triangle) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{0, 0}, {3, 0}, {1, 2}};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{0, 0}, {3, 0}, {1, 2}};

  shulpin_tbb_test_module::MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_tbb, octagone) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{1, 0}, {2, 0}, {3, 1}, {3, 2}, {2, 3}, {1, 3}, {0, 2}, {0, 1}};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{0, 1}, {1, 0}, {2, 0}, {3, 1}, {3, 2}, {2, 3}, {1, 3}, {0, 2}};

  shulpin_tbb_test_module::MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_tbb, repeated_points) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {2, 0}, {0, 0}};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};

  shulpin_tbb_test_module::MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_tbb, real_case) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{1, 1}, {3, 2}, {5, 1}, {4, 3}, {2, 4}, {1, 3}, {3, 3}};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{1, 1}, {5, 1}, {4, 3}, {2, 4}, {1, 3}};

  shulpin_tbb_test_module::MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_tbb, star_case) {
  // clang-format off
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{0.0, 3.0},
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
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{-3.0, 1.0},
      {-2.5, -3.0},
      {2.5, -3.0},
      {3.0, 1.0},
      {0.0, 3.0},
  };
  // clang-format on
  shulpin_tbb_test_module::MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_tbb, one_point_validation_false) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{0, 0}};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{0, 0}};

  shulpin_tbb_test_module::TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_tbb, three_points_validation_false) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{1, 1}, {2, 2}};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{1, 1}, {2, 2}};

  shulpin_tbb_test_module::TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_tbb, zero_points_validation_false) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {};

  shulpin_tbb_test_module::TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_tbb, circle_r10_p100) {
  shulpin_i_jarvis_tbb::Point center{0, 0};

  double radius = 10.0;
  size_t num_points = 100;

  std::vector<shulpin_i_jarvis_tbb::Point> input =
      shulpin_tbb_test_module::GeneratePointsInCircle(num_points, center, radius);
  std::vector<shulpin_i_jarvis_tbb::Point> expected = input;

  shulpin_tbb_test_module::TestBodyRandomCircle(input, expected, num_points);
}

TEST(shulpin_i_jarvis_tbb, circle_r10_p200) {
  shulpin_i_jarvis_tbb::Point center{0, 0};

  double radius = 10.0;
  size_t num_points = 200;

  std::vector<shulpin_i_jarvis_tbb::Point> input =
      shulpin_tbb_test_module::GeneratePointsInCircle(num_points, center, radius);
  std::vector<shulpin_i_jarvis_tbb::Point> expected = input;

  shulpin_tbb_test_module::TestBodyRandomCircle(input, expected, num_points);
}

TEST(shulpin_i_jarvis_tbb, random_10_points) {
  size_t num_points = 10;

  std::vector<shulpin_i_jarvis_tbb::Point> input = shulpin_tbb_test_module::GenerateRandomPoints(num_points);
  std::vector<shulpin_i_jarvis_tbb::Point> expected = shulpin_tbb_test_module::ComputeConvexHull(input);

  shulpin_tbb_test_module::RandomTestBody(input, expected);
}

TEST(shulpin_i_jarvis_tbb, random_50_points) {
  size_t num_points = 50;

  std::vector<shulpin_i_jarvis_tbb::Point> input = shulpin_tbb_test_module::GenerateRandomPoints(num_points);
  std::vector<shulpin_i_jarvis_tbb::Point> expected = shulpin_tbb_test_module::ComputeConvexHull(input);

  shulpin_tbb_test_module::RandomTestBody(input, expected);
}

TEST(shulpin_i_jarvis_tbb, random_100_points) {
  size_t num_points = 100;

  std::vector<shulpin_i_jarvis_tbb::Point> input = shulpin_tbb_test_module::GenerateRandomPoints(num_points);
  std::vector<shulpin_i_jarvis_tbb::Point> expected = shulpin_tbb_test_module::ComputeConvexHull(input);

  shulpin_tbb_test_module::RandomTestBody(input, expected);
}
