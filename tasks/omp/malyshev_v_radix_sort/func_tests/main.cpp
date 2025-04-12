#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/malyshev_v_radix_sort/include/ops_omp.hpp"

TEST(malyshev_v_radix_sort_omp, ordinary_test) {
  std::vector<double> input_vector = {3.4, 1.2, 0.5, 7.8, 2.3, 4.5, 6.7, 8.9, 1.0, 0.2, 5.6, 4.3, 9.1, 1.5, 3.0};
  std::vector<double> out(input_vector.size(), 0.0);
  std::vector<double> sorted_vector = {0.2, 0.5, 1.0, 1.2, 1.5, 2.3, 3.0, 3.4, 4.3, 4.5, 5.6, 6.7, 7.8, 8.9, 9.1};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  malyshev_v_radix_sort_omp::RadixSortDoubleOMP task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(out, sorted_vector);
}

TEST(malyshev_v_radix_sort_omp, random_vector_test) {
  constexpr size_t kSize = 50;
  std::vector<double> input_vector(kSize);
  for (auto& val : input_vector) {
    val = (rand() % 2000 - 1000) / 10.0;
  }
  std::vector<double> out(kSize, 0.0);
  std::vector<double> reference = input_vector;
  std::ranges::sort(reference);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  malyshev_v_radix_sort_omp::RadixSortDoubleOMP task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(out, reference);
}

TEST(malyshev_v_radix_sort_omp, negative_numbers_test) {
  std::vector<double> input_vector = {-5.4, -2.3, -9.1, -0.5, -3.7, -1.2};
  std::vector<double> out(input_vector.size(), 0.0);
  std::vector<double> sorted_vector = {-9.1, -5.4, -3.7, -2.3, -1.2, -0.5};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  malyshev_v_radix_sort_omp::RadixSortDoubleOMP task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(out, sorted_vector);
}

TEST(malyshev_v_radix_sort_omp, zeros_test) {
  std::vector<double> input_vector = {0.0, -0.0, 0.0, 0.0, -0.0};
  std::vector<double> out(input_vector.size(), 0.0);
  std::vector<double> sorted_vector = {-0.0, -0.0, 0.0, 0.0, 0.0};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  malyshev_v_radix_sort_omp::RadixSortDoubleOMP task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(out, sorted_vector);
}

TEST(malyshev_v_radix_sort_omp, duplicates_test) {
  std::vector<double> input_vector = {3.3, 1.1, 2.2, 3.3, 1.1, 2.2};
  std::vector<double> out(input_vector.size(), 0.0);
  std::vector<double> sorted_vector = {1.1, 1.1, 2.2, 2.2, 3.3, 3.3};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  malyshev_v_radix_sort_omp::RadixSortDoubleOMP task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(out, sorted_vector);
}

TEST(malyshev_v_radix_sort_omp, reverse_order_test) {
  std::vector<double> input_vector = {5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
  std::vector<double> out(input_vector.size(), 0.0);
  std::vector<double> sorted_vector = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  malyshev_v_radix_sort_omp::RadixSortDoubleOMP task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(out, sorted_vector);
}

TEST(malyshev_v_radix_sort_omp, validation_fail_test) {
  std::vector<double> input_vector(10, 1.0);
  std::vector<double> out(15, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  malyshev_v_radix_sort_omp::RadixSortDoubleOMP task(task_data);
  ASSERT_FALSE(task.Validation());
}