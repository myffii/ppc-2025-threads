#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "../include/ops_omp.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<double> RandomVector(size_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dist(-5000, 8000);
  std::vector<double> vec(size);
  std::ranges::generate(vec, [&dist, &gen] { return dist(gen); });
  return vec;
}

void STest(std::vector<double> in) {
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_omp::TestTaskParallelOmp(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

void STest(size_t size) { STest(RandomVector(size)); }
}  // namespace

TEST(petrov_a_radix_double_batcher_omp, test_0) { STest(0); }

TEST(petrov_a_radix_double_batcher_omp, test_1) { STest(1); }

TEST(petrov_a_radix_double_batcher_omp, test_2) { STest(2); }

TEST(petrov_a_radix_double_batcher_omp, test_3) { STest(3); }

TEST(petrov_a_radix_double_batcher_omp, test_4) { STest(4); }

TEST(petrov_a_radix_double_batcher_omp, test_5) { STest(5); }

TEST(petrov_a_radix_double_batcher_omp, test_6) { STest(6); }

TEST(petrov_a_radix_double_batcher_omp, test_7) { STest(7); }

TEST(petrov_a_radix_double_batcher_omp, test_8) { STest(8); }

TEST(petrov_a_radix_double_batcher_omp, test_9) { STest(9); }

TEST(petrov_a_radix_double_batcher_omp, test_10) { STest(10); }

TEST(petrov_a_radix_double_batcher_omp, test_11) { STest(11); }

TEST(petrov_a_radix_double_batcher_omp, test_111) { STest(111); }

TEST(petrov_a_radix_double_batcher_omp, test_213) { STest(213); }

TEST(petrov_a_radix_double_batcher_omp, test_already_sorted) { STest({1, 2, 3, 4, 5, 10, 15, 16, 100}); }

TEST(petrov_a_radix_double_batcher_omp, test_same) { STest(std::vector<double>(81, 555)); }