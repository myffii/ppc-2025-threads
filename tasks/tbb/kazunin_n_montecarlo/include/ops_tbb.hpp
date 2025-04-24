#pragma once

#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <random>
#include <utility>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

namespace kazunin_n_montecarlo_tbb {

template <std::size_t N, typename F>
class MonteCarloTbb : public ppc::core::Task {
 public:
  explicit MonteCarloTbb(ppc::core::TaskDataPtr task_data, F f) : Task(std::move(task_data)), f_(f) {}

  bool ValidationImpl() override {
    return task_data->inputs_count[0] == 1 && task_data->inputs_count[1] == N && task_data->outputs_count[0] == 1 &&
           (*reinterpret_cast<std::size_t*>(task_data->inputs[0]) > 0);
  }

  bool PreProcessingImpl() override {
    precision_ = *reinterpret_cast<std::size_t*>(task_data->inputs[0]);
    limits_ = *reinterpret_cast<decltype(limits_)*>(task_data->inputs[1]);
    std::ranges::generate(generators_, [&, i = 0]() mutable { return MakeGenerator(limits_[i++]); });
    total_space_ = std::accumulate(
        limits_.begin(), limits_.end(), 1.0,
        [](const double acc, const std::pair<double, double>& limit) { return acc * (limit.second - limit.first); });
    return true;
  }
  bool RunImpl() override {
    oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
    double total_sum = arena.execute([&] {
      return oneapi::tbb::parallel_reduce(
          oneapi::tbb::blocked_range<std::size_t>(0, precision_, precision_ / arena.max_concurrency()), 0.0,
          [&](const tbb::blocked_range<std::size_t>& range, double sum) {
            std::array<double, N> random_args;
            const auto generators = generators_;
            for (std::size_t iter = range.begin(); iter < range.end(); ++iter) {
              std::ranges::generate(random_args, [&generators, j = 0]() mutable { return generators[j++](); });
              sum += f_(random_args);
            }
            return sum;
          },
          std::plus<>());
    });

    result_ = (total_space_ * total_sum) / precision_;
    return true;
  }
  bool PostProcessingImpl() override {
    *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
    return true;
  }

 private:
  std::function<double()> MakeGenerator(const std::pair<double, double>& interval) {
    std::uniform_real_distribution<double> distrib(interval.first, interval.second);
    std::mt19937 engine(seedgen_());
    return [distrib, engine]() mutable { return distrib(engine); };
  }

  F f_;
  std::size_t precision_;
  double total_space_;
  std::array<std::pair<double, double>, N> limits_;
  std::array<std::function<double()>, N> generators_;
  std::random_device seedgen_;
  double result_;
};

}  // namespace kazunin_n_montecarlo_tbb

namespace kazunin_n_montecarlo_seq {

template <std::size_t N, typename F>
class MonteCarloSeq : public ppc::core::Task {
 public:
  explicit MonteCarloSeq(ppc::core::TaskDataPtr task_data, F f) : Task(std::move(task_data)), f_(f) {}

  bool ValidationImpl() override {
    return task_data->inputs_count[0] == 1 && task_data->inputs_count[1] == N && task_data->outputs_count[0] == 1 &&
           (*reinterpret_cast<std::size_t*>(task_data->inputs[0]) > 0);
  }

  bool PreProcessingImpl() override {
    precision_ = *reinterpret_cast<std::size_t*>(task_data->inputs[0]);
    limits_ = *reinterpret_cast<decltype(limits_)*>(task_data->inputs[1]);
    std::ranges::generate(generators_, [&, i = 0]() mutable { return MakeGenerator(limits_[i++]); });
    total_space_ = std::accumulate(
        limits_.begin(), limits_.end(), 1.0,
        [](const double acc, const std::pair<double, double>& limit) { return acc * (limit.second - limit.first); });
    return true;
  }
  bool RunImpl() override {
    double sum = 0.0;

    std::array<double, N> random_args;
    for (std::size_t iter = 0; iter < precision_; ++iter) {
      std::ranges::generate(random_args, [this, j = 0]() mutable { return generators_[j++](); });
      sum += f_(random_args);
    }

    result_ = (total_space_ * sum) / precision_;
    return true;
  }
  bool PostProcessingImpl() override {
    *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
    return true;
  }

 private:
  std::function<double()> MakeGenerator(const std::pair<double, double>& interval) {
    std::uniform_real_distribution<double> distrib(interval.first, interval.second);
    std::mt19937 engine(seedgen_());
    return [distrib, engine]() mutable { return distrib(engine); };
  }

  F f_;
  std::size_t precision_;
  double total_space_;
  std::array<std::pair<double, double>, N> limits_;
  std::array<std::function<double()>, N> generators_;
  std::random_device seedgen_;
  double result_;
};

}  // namespace kazunin_n_montecarlo_seq