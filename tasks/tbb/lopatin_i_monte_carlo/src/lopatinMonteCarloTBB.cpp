#include "tbb/lopatin_i_monte_carlo/include/lopatinMonteCarloTBB.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <functional>
#include <random>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/parallel_reduce.h"
#include "oneapi/tbb/task_arena.h"

namespace lopatin_i_monte_carlo_tbb {

bool TestTaskTBB::ValidationImpl() {
  const bool outputs_valid = !task_data->outputs_count.empty() && task_data->outputs_count[0] == 1;
  const bool inputs_valid = task_data->inputs_count.size() == 2 &&
                            (task_data->inputs_count[0] % 2 == 0) &&  // odd num of bounds
                            task_data->inputs_count[1] == 1;          // iterations count

  auto* iter_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
  const int iterations = *iter_ptr;
  const bool iter_valid = iterations > 0;

  bool bounds_valid = true;  // bounds
  auto* bounds_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  if (bounds_ptr[0] >= bounds_ptr[1]) {
    bounds_valid = false;
  }

  return outputs_valid && inputs_valid && iter_valid && bounds_valid;
}

bool TestTaskTBB::PreProcessingImpl() {
  auto* bounds_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  size_t bounds_size = task_data->inputs_count[0];
  integrationBounds_.resize(bounds_size);
  std::copy(bounds_ptr, bounds_ptr + bounds_size, integrationBounds_.begin());

  auto* iter_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
  iterations_ = *iter_ptr;
  return true;
}

bool TestTaskTBB::RunImpl() {
  const size_t d = integrationBounds_.size() / 2;  // dimensions

  // init random numbers generator
  std::random_device rd;
  std::seed_seq seed{rd(), static_cast<unsigned int>(std::time(nullptr))};
  const size_t num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::mt19937::result_type> seeds(num_threads);
  seed.generate(seeds.begin(), seeds.end());

  // volume of integration region
  double volume = 1.0;
  for (size_t j = 0; j < d; ++j) {
    volume *= (integrationBounds_[(2 * j) + 1] - integrationBounds_[2 * j]);
  }

  // tbb parallel reduction
  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  double total_sum = arena.execute([&] {
    return oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<std::size_t>(0, iterations_, iterations_ / arena.max_concurrency()), 0.0,
        [&](const oneapi::tbb::blocked_range<std::size_t>& range, double sum) {
          static thread_local std::mt19937 local_rnd;
          static thread_local bool initialized = false;
          if (!initialized) {
            const size_t idx = reinterpret_cast<uintptr_t>(&local_rnd) % num_threads;
            local_rnd.seed(seeds[idx]);
            initialized = true;
          }
          std::uniform_real_distribution<> dis(0.0, 1.0);
          std::vector<double> point(d);

          for (size_t i = range.begin(); i < range.end(); ++i) {
            for (size_t j = 0; j < d; ++j) {
              const double min = integrationBounds_[2 * j];
              const double max = integrationBounds_[(2 * j) + 1];
              point[j] = min + (max - min) * dis(local_rnd);
            }
            sum += integrand_(point);
          }
          return sum;
        },
        std::plus<>());
  });

  result_ = (total_sum / iterations_) * volume;
  return true;
}

bool TestTaskTBB::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  *output_ptr = result_;
  return true;
}

}  // namespace lopatin_i_monte_carlo_tbb