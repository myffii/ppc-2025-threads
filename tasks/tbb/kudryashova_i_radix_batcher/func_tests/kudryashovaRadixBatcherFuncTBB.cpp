#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/kudryashova_i_radix_batcher/include/kudryashovaRadixBatcherTBB.hpp"

std::vector<double> kudryashova_i_radix_batcher_tbb::GetRandomDoubleVector(int size) {
  std::vector<double> vector(size);
  std::mt19937 generator(static_cast<unsigned>(std::time(nullptr)));
  std::uniform_real_distribution<double> distribution(-1000.0, 1000.0);
  for (int i = 0; i < size; ++i) {
    vector[i] = distribution(generator);
  }
  return vector;
}

TEST(kudryashova_i_radix_batcher_tbb, tbb_radix_test_1) {
  int global_vector_size = 3;
  std::vector<double> global_vector = {5.69, -2.11, 0.52};
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_tbb::TestTaskTBB task_tbb(task_data);
  ASSERT_TRUE(task_tbb.ValidationImpl());
  task_tbb.PreProcessingImpl();
  task_tbb.RunImpl();
  task_tbb.PostProcessingImpl();
  std::vector<double> sort_global_vector = global_vector;
  std::ranges::sort(sort_global_vector);
  ASSERT_EQ(result, sort_global_vector);
}

TEST(kudryashova_i_radix_batcher_tbb, tbb_radix_test_2) {
  int global_vector_size = 11;
  std::vector<double> global_vector = {-8.55,   1.85,   -4.0,   2.81828, 8.77,   -5.56562,
                                       -15.823, -6.971, 3.1615, 0.0,     10.1415};
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_tbb::TestTaskTBB task_tbb(task_data);
  ASSERT_TRUE(task_tbb.ValidationImpl());
  task_tbb.PreProcessingImpl();
  task_tbb.RunImpl();
  task_tbb.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}

TEST(kudryashova_i_radix_batcher_tbb, tbb_radix_all_equal_elem_test_3) {
  int global_vector_size = 8;
  std::vector<double> global_vector = {1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57};
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_tbb::TestTaskTBB task_tbb(task_data);
  ASSERT_TRUE(task_tbb.ValidationImpl());
  task_tbb.PreProcessingImpl();
  task_tbb.RunImpl();
  task_tbb.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}

TEST(kudryashova_i_radix_batcher_tbb, tbb_radix_all_zero_elem_test_4) {
  int global_vector_size = 7;
  std::vector<double> global_vector = {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00};
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_tbb::TestTaskTBB task_tbb(task_data);
  ASSERT_TRUE(task_tbb.ValidationImpl());
  task_tbb.PreProcessingImpl();
  task_tbb.RunImpl();
  task_tbb.PostProcessingImpl();
  ASSERT_EQ(result, global_vector);
}

TEST(kudryashova_i_radix_batcher_tbb, tbb_radix_test_5) {
  int global_vector_size = 8;
  std::vector<double> global_vector = {1.852, 1.2, 1.7852, 5.57, 4.525, 100.769, 75.123, 85.57};
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_tbb::TestTaskTBB task_tbb(task_data);
  ASSERT_TRUE(task_tbb.ValidationImpl());
  task_tbb.PreProcessingImpl();
  task_tbb.RunImpl();
  task_tbb.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}

TEST(kudryashova_i_radix_batcher_tbb, tbb_radix_random_test_1) {
  int global_vector_size = 18;
  std::vector<double> global_vector = kudryashova_i_radix_batcher_tbb::GetRandomDoubleVector(global_vector_size);
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_tbb::TestTaskTBB task_tbb(task_data);
  ASSERT_TRUE(task_tbb.ValidationImpl());
  task_tbb.PreProcessingImpl();
  task_tbb.RunImpl();
  task_tbb.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}

TEST(kudryashova_i_radix_batcher_tbb, tbb_radix_random_test_2) {
  int global_vector_size = 512;
  std::vector<double> global_vector = kudryashova_i_radix_batcher_tbb::GetRandomDoubleVector(global_vector_size);
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_tbb::TestTaskTBB task_tbb(task_data);
  ASSERT_TRUE(task_tbb.ValidationImpl());
  task_tbb.PreProcessingImpl();
  task_tbb.RunImpl();
  task_tbb.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}

TEST(kudryashova_i_radix_batcher_tbb, tbb_radix_test_regular_order) {
  int global_vector_size = 100;
  std::vector<double> global_vector = kudryashova_i_radix_batcher_tbb::GetRandomDoubleVector(global_vector_size);
  std::ranges::sort(global_vector);
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_tbb::TestTaskTBB task_tbb(task_data);
  ASSERT_TRUE(task_tbb.ValidationImpl());
  task_tbb.PreProcessingImpl();
  task_tbb.RunImpl();
  task_tbb.PostProcessingImpl();
  ASSERT_EQ(result, global_vector);
}

TEST(kudryashova_i_radix_batcher_tbb, tbb_radix_test_reverse_order) {
  int global_vector_size = 100;
  std::vector<double> global_vector = kudryashova_i_radix_batcher_tbb::GetRandomDoubleVector(global_vector_size);
  std::ranges::sort(global_vector, std::greater<>());
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_tbb::TestTaskTBB task_tbb(task_data);
  ASSERT_TRUE(task_tbb.ValidationImpl());
  task_tbb.PreProcessingImpl();
  task_tbb.RunImpl();
  task_tbb.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}
