
#include <gtest/gtest.h>
#include <tbb/global_control.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/vladimirova_j_m_monte_karlo/include/ops_tbb.hpp"

namespace {
bool PiVal314(std::vector<double> arr, size_t size = 2) {
  double x = arr[0];
  double y = arr[1];
  return (((x * x) + (y * y)) <= 1);
}  // namespace

bool SphereR3(std::vector<double> arr, size_t size = 3) {
  double x = arr[0];
  double y = arr[1];
  double z = arr[2];
  return (((x * x) + (y * y) + (z * z)) <= 9);
};

bool Rectangle34(std::vector<double> arr, size_t size = 2) {
  double x = arr[0];
  double y = arr[1];
  return ((x >= 0) && (x <= 3) && (y >= 0) && (y <= 4));
}

bool Parallelogram333(std::vector<double> arr, size_t size = 3) {
  double x = arr[0];
  double y = arr[1];
  double z = arr[2];
  return ((x >= 0) && (x <= 3) && (y >= 0) && (y <= 3) && (z >= 0) && (z <= 3));
}

bool BigParallelogram100100100(std::vector<double> arr, size_t size = 3) {
  double x = arr[0];
  double y = arr[1];
  double z = arr[2];
  return ((x >= 0) && (x <= 100) && (y >= 0) && (y <= 100) && (z >= 0) && (z <= 100));
}

bool Parabola(std::vector<double> arr, size_t size = 2) {
  double x = arr[0];
  double y = arr[1];
  return (((3 * x * x) + (12 * x) - 10) <= y);
}

bool HypercubeX4Pr4433(std::vector<double> arr, size_t size = 4) {
  double x = arr[0];
  double y = arr[1];
  double z = arr[2];
  double w = arr[3];

  return ((x >= 0) && (x <= 4) && (y >= 0) && (y <= 4) && (z >= 0) && (z <= 3) && (w >= 0) && (w <= 3));
}

bool TriangleModuleMinus5(std::vector<double> arr, size_t size = 2) {
  double x = arr[0];
  double y = arr[1];
  if (x > 0) {
    x *= -1;
  }
  return x <= (y - 5);
}

bool SomeRandomFunc(std::vector<double> arr, size_t size = 2) {
  double x = arr[0];
  double y = arr[1];
  return (((x * x) - (1.2 * x * x * x) - y - 5) <= 0);
}

}  // namespace

TEST(vladimirova_j_m_monte_karlo_tbb, test_zero_var) {
  std::vector<double> val_b = {};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(PiVal314));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(70000));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), false);
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_one_var) {
  std::vector<double> val_b = {-1, 1};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(PiVal314));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(70000));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), false);
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_no_limit_var) {
  std::vector<double> val_b = {-1, 1, 0};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(PiVal314));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(70000));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), false);
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_zero_size_limit_var) {
  std::vector<double> val_b = {-1, 1, 0, 0};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(PiVal314));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(70000));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), false);
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_one_point) {
  std::vector<double> val_b = {-1, 1, 0, 1};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(PiVal314));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(1));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  ASSERT_TRUE((2 == (int)out[0]) || ((int)out[0] == 0));
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_PiVal_314) {
  std::vector<double> val_b = {-1, 1, -1, 1};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(PiVal314));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(9000));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  ASSERT_EQ(3, (int)out[0]);
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_PiVal_314_2) {
  std::vector<double> val_b = {-1, 1, -1, 1};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(PiVal314));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(50000));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  ASSERT_TRUE((3.14 - out[0]) < 0.5);
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_PiVal_314_1_2) {
  std::vector<double> val_b = {0, 1, -1, 1};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(PiVal314));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(50000));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  ASSERT_TRUE(((3.14 / 2) - out[0]) < 0.5);
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_SphereR3_1) {
  std::vector<double> val_b = {-3, 3, -3, 3, -3, 3};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(SphereR3));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(50000));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  ASSERT_TRUE((113 - out[0]) < 6);
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_SphereR3_1_2) {
  std::vector<double> val_b = {0, 3, -3, 3, -3, 3};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(SphereR3));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(9000));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  ASSERT_TRUE(((113.0 / 2.0) - (int)out[0]) < 5);
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_SphereR3_1_9) {
  std::vector<double> val_b = {0, 3, 0, 3, 0, 3};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(SphereR3));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(10000));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  ASSERT_TRUE(((113.0 / 9.0) - out[0]) < 5);
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_Rectangle34) {
  std::vector<double> val_b = {0, 3, 0, 4};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(Rectangle34));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(30000));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  ASSERT_TRUE(((3 * 4) - out[0]) < 2);
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_Rectangle34_2) {
  std::vector<double> val_b = {0, 4, 0, 4};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(Rectangle34));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(30000));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  ASSERT_TRUE(((3 * 4) - out[0]) < 1);
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_Parallelogram333) {
  std::vector<double> val_b = {0, 3, 0, 3, 0, 3};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(Parallelogram333));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(900));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  ASSERT_TRUE(((3 * 3 * 3) - out[0]) == 0);
}
TEST(vladimirova_j_m_monte_karlo_tbb, test_BigParallelogram100100100) {
  std::vector<double> val_b = {0, 100, 0, 100, 0, 100};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(BigParallelogram100100100));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(900));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  ASSERT_TRUE(((100 * 100 * 100) - (int)out[0]) == 0);
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_Parabola_1_5_2_9) {
  std::vector<double> val_b = {-4, 0, -24, 0};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(Parabola));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(50000));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  ASSERT_TRUE((72 - out[0]) < 3);
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_HypercubeX4Pr4433) {
  std::vector<double> val_b = {0, 4, 0, 4, 0, 4, 0, 4};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(HypercubeX4Pr4433));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(50000));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  ASSERT_TRUE((3 * 3 * 4 * 4 - out[0]) < 5);
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_TriangleModuleMinus5) {
  std::vector<double> val_b = {-5, 5, 0, 5};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(TriangleModuleMinus5));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(50000));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  ASSERT_TRUE((25 - out[0]) < 1);
}

TEST(vladimirova_j_m_monte_karlo_tbb, test_SomeRandomFunc) {
  std::vector<double> val_b = {-2, 0, -5.5, -2};
  std::vector<double> out(1, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(val_b.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(SomeRandomFunc));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(10000));
  task_data_tbb->inputs_count.emplace_back(val_b.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  vladimirova_j_m_monte_karlo_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  ASSERT_TRUE((2.533 - out[0]) < 1);
}
