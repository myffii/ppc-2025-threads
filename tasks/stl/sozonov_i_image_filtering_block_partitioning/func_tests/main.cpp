#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/sozonov_i_image_filtering_block_partitioning/include/ops_stl.hpp"

namespace sozonov_i_image_filtering_block_partitioning_stl {

std::vector<double> ZeroEdges(std::vector<double> img, int wdth, int hght) {
  for (int i = 0; i < wdth; ++i) {
    img[i] = 0;
    img[((hght - 1) * wdth) + i] = 0;
  }
  for (int i = 1; i < hght - 1; ++i) {
    img[i * wdth] = 0;
    img[(i * wdth) + wdth - 1] = 0;
  }

  return img;
}
}  // namespace sozonov_i_image_filtering_block_partitioning_stl

TEST(sozonov_i_image_filtering_block_partitioning_stl, test_empty_image) {
  const int width = 0;
  const int height = 0;

  // Create data
  std::vector<double> in;
  std::vector<double> out;

  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->inputs_count.emplace_back(width);
  task_data_stl->inputs_count.emplace_back(height);
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_block_partitioning_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_FALSE(test_task_stl.Validation());
}

TEST(sozonov_i_image_filtering_block_partitioning_stl, test_image_less_than_3x3) {
  const int width = 2;
  const int height = 2;

  // Create data
  std::vector<double> in = {4, 6, 8, 24};
  std::vector<double> out(width * height, 0);

  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->inputs_count.emplace_back(width);
  task_data_stl->inputs_count.emplace_back(height);
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_block_partitioning_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_FALSE(test_task_stl.Validation());
}

TEST(sozonov_i_image_filtering_block_partitioning_stl, test_wrong_pixels) {
  const int width = 5;
  const int height = 3;

  // Create data
  std::vector<double> in = {143, 6, 853, -24, 31, -25, 1, 5, -7, 361, 28, 98, -45, 982, 461};
  std::vector<double> out(width * height, 0);

  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->inputs_count.emplace_back(width);
  task_data_stl->inputs_count.emplace_back(height);
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_block_partitioning_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_FALSE(test_task_stl.Validation());
}

TEST(sozonov_i_image_filtering_block_partitioning_stl, test_3x3) {
  const int width = 3;
  const int height = 3;

  // Create data
  std::vector<double> in = {4, 6, 8, 24, 31, 25, 1, 5, 7};
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0, 0, 0, 0, 16.5, 0, 0, 0, 0};

  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->inputs_count.emplace_back(width);
  task_data_stl->inputs_count.emplace_back(height);
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_block_partitioning_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  EXPECT_EQ(out, ans);
}

TEST(sozonov_i_image_filtering_block_partitioning_stl, test_5x3) {
  const int width = 5;
  const int height = 3;

  // Create data
  std::vector<double> in = {34, 24, 27, 67, 42, 48, 93, 26, 47, 2, 34, 13, 81, 24, 32};
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0, 0, 0, 0, 0, 0, 48.125, 45.5, 38, 0, 0, 0, 0, 0, 0};

  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->inputs_count.emplace_back(width);
  task_data_stl->inputs_count.emplace_back(height);
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_block_partitioning_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  EXPECT_EQ(out, ans);
}

TEST(sozonov_i_image_filtering_block_partitioning_stl, test_5x5) {
  const int width = 5;
  const int height = 5;

  // Create data
  std::vector<double> in(width * height);
  std::iota(in.begin(), in.end(), 0);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0, 0, 0, 0, 0, 0, 6, 7, 8, 0, 0, 11, 12, 13, 0, 0, 16, 17, 18, 0, 0, 0, 0, 0, 0};

  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->inputs_count.emplace_back(width);
  task_data_stl->inputs_count.emplace_back(height);
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_block_partitioning_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  EXPECT_EQ(out, ans);
}

TEST(sozonov_i_image_filtering_block_partitioning_stl, test_5x7) {
  const int width = 5;
  const int height = 7;

  // Create data
  std::vector<double> in(width * height);
  std::iota(in.begin(), in.end(), 0);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0,  0, 0, 0,  0,  0,  6, 7, 8,  0,  0,  11, 12, 13, 0, 0, 16, 17,
                             18, 0, 0, 21, 22, 23, 0, 0, 26, 27, 28, 0,  0,  0,  0, 0, 0};

  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->inputs_count.emplace_back(width);
  task_data_stl->inputs_count.emplace_back(height);
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_block_partitioning_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  EXPECT_EQ(out, ans);
}

TEST(sozonov_i_image_filtering_block_partitioning_stl, test_10x4) {
  const int width = 10;
  const int height = 4;

  // Create data
  std::vector<double> in(width * height);
  std::iota(in.begin(), in.end(), 0);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0, 0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 11, 12, 13, 14, 15, 16, 17, 18, 0,
                             0, 21, 22, 23, 24, 25, 26, 27, 28, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0};

  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->inputs_count.emplace_back(width);
  task_data_stl->inputs_count.emplace_back(height);
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_block_partitioning_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  EXPECT_EQ(out, ans);
}

TEST(sozonov_i_image_filtering_block_partitioning_stl, test_100x100) {
  const int width = 100;
  const int height = 100;

  // Create data
  std::vector<double> in(width * height, 1);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans(width * height, 1);

  ans = sozonov_i_image_filtering_block_partitioning_stl::ZeroEdges(ans, width, height);

  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->inputs_count.emplace_back(width);
  task_data_stl->inputs_count.emplace_back(height);
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_block_partitioning_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  EXPECT_EQ(out, ans);
}

TEST(sozonov_i_image_filtering_block_partitioning_stl, test_150x100) {
  const int width = 150;
  const int height = 100;

  // Create data
  std::vector<double> in(width * height, 1);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans(width * height, 1);

  ans = sozonov_i_image_filtering_block_partitioning_stl::ZeroEdges(ans, width, height);

  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->inputs_count.emplace_back(width);
  task_data_stl->inputs_count.emplace_back(height);
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_block_partitioning_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  EXPECT_EQ(out, ans);
}

TEST(sozonov_i_image_filtering_block_partitioning_stl, test_120x200) {
  const int width = 120;
  const int height = 200;

  // Create data
  std::vector<double> in(width * height, 1);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans(width * height, 1);

  ans = sozonov_i_image_filtering_block_partitioning_stl::ZeroEdges(ans, width, height);

  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->inputs_count.emplace_back(width);
  task_data_stl->inputs_count.emplace_back(height);
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_block_partitioning_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  EXPECT_EQ(out, ans);
}
