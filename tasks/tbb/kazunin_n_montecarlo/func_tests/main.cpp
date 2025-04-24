#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/kazunin_n_montecarlo/include/ops_tbb.hpp"

using std::sin;

namespace {
template <std::size_t N, typename F>
void MonteCarloTest(F f, std::size_t precision, std::array<std::pair<double, double>, N> limits) {
  double out = 0.0;
  double ref = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t *>(&precision), reinterpret_cast<uint8_t *>(&limits)};
  task_data->inputs_count = {1, N};
  task_data->outputs = {reinterpret_cast<uint8_t *>(&out)};
  task_data->outputs_count = {1};

  // Create Task
  kazunin_n_montecarlo_tbb::MonteCarloTbb<N, F> task_tbb(task_data, f);
  ASSERT_TRUE(task_tbb.Validation());
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();

  task_data->outputs = {reinterpret_cast<uint8_t *>(&ref)};
  kazunin_n_montecarlo_seq::MonteCarloSeq<N, F> task_seq(task_data, f);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();
  EXPECT_NEAR(out, ref, 0.2);
}
template <std::size_t N, typename F>
void InvalidMonteCarloTest(F f, std::size_t precision, std::array<std::pair<double, double>, N> limits) {
  double out = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t *>(&precision), reinterpret_cast<uint8_t *>(&limits)};
  task_data->inputs_count = {1, N};
  task_data->outputs = {reinterpret_cast<uint8_t *>(&out)};
  task_data->outputs_count = {1};

  // Create Task
  kazunin_n_montecarlo_tbb::MonteCarloTbb<N, F> task(task_data, f);
  EXPECT_FALSE(task.Validation());
}
}  // namespace

TEST(kazunin_n_montecarlo_tbb, zero_precision) {
  const std::size_t n = 2;
  InvalidMonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0,
                               [](const double acc, double component) { return acc * sin(component); });
      },
      0, {{{0.0, 1.0}, {-1.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_tbb, no_bounds) {
  const std::size_t n = 2;
  InvalidMonteCarloTest<n>([](const std::array<double, n> &args) { return -1; }, 0, {});
}

TEST(kazunin_n_montecarlo_tbb, sin_prod_2d) {
  const std::size_t n = 2;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0,
                               [](const double acc, double component) { return acc * sin(component); });
      },
      5000, {{{0.0, 1.0}, {-1.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_tbb, sin_sum_2d) {
  const std::size_t n = 2;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 0.0,
                               [](const double acc, double component) { return acc + sin(component); });
      },
      7000, {{{0.0, 1.0}, {-1.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_tbb, sin_prod_3d) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0,
                               [](const double acc, double component) { return acc * sin(component); });
      },
      5000, {{{0.0, 1.0}, {-1.0, 0.0}, {-1.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_tbb, sin_sum_3d) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 0.0,
                               [](const double acc, double component) { return acc + sin(component); });
      },
      5000, {{{0.0, 1.0}, {-1.0, 0.0}, {-1.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_tbb, product_3d) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0, std::multiplies<>());
      },
      5000, {{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
}

TEST(kazunin_n_montecarlo_tbb, sum_3d) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0, std::multiplies<>());
      },
      5000, {{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
}

TEST(kazunin_n_montecarlo_tbb, sin_prod_2d_degenerate) {
  const std::size_t n = 2;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0,
                               [](const double acc, double component) { return acc * sin(component); });
      },
      5000, {{{0.0, 0.0}, {0.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_tbb, sin_sum_2d_degenerate) {
  const std::size_t n = 2;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 0.0,
                               [](const double acc, double component) { return acc + sin(component); });
      },
      5000, {{{0.0, 0.0}, {0.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_tbb, sin_prod_3d_degenerate) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0,
                               [](const double acc, double component) { return acc * sin(component); });
      },
      5000, {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_tbb, sin_sum_3d_degenerate) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 0.0,
                               [](const double acc, double component) { return acc + sin(component); });
      },
      5000, {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_tbb, product_3d_degenerate) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0, std::multiplies<>());
      },
      5000, {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_tbb, sum_3d_degenerate) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0, std::multiplies<>());
      },
      5000, {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_tbb, sum_3d_coinciding_bounds) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0, std::multiplies<>());
      },
      5000, {{{1.0, 1.0}, {5.0, 5.0}, {9.0, 9.0}}});
}
