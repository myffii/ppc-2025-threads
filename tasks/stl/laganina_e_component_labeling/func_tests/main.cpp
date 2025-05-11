#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/laganina_e_component_labeling/include/ops_stl.hpp"
TEST(laganina_e_component_labeling_stl, validation_test1) {
  int m = 0;
  int n = 1;
  // Create data
  std::vector<int> in(m * n, 0);
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task..
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), false);
}

TEST(laganina_e_component_labeling_stl, validation_test4) {
  int m = 1;
  int n = 0;
  // Create data
  std::vector<int> in(m * n, 0);
  std::vector<int> out(m * n, 0);
  // Create task_data 2
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);
  task_data_omp->outputs_count.emplace_back(n);
  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), false);
}

TEST(laganina_e_component_labeling_stl, validation_test2) {
  int m = 0;
  int n = 0;
  // Create data
  std::vector<int> in(m * n, 3);
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), false);
}

TEST(laganina_e_component_labeling_stl, validation_test3) {
  int m = 3;
  int n = 1;
  // Create data
  std::vector<int> in(m * n, 3);
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), false);
}

TEST(laganina_e_component_labeling_stl, Find_test) {
  int m = 3;
  int n = 3;
  // Create data
  std::vector<int> in = {1, 0, 1, 1, 1, 0, 0, 1, 1};
  std::vector<int> out(m * n, 0);
  std::vector<int> exp_out = {1, 0, 2, 1, 1, 0, 0, 1, 1};
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(out, exp_out);
}

TEST(laganina_e_component_labeling_stl, all_one) {
  int m = 3;
  int n = 2;
  // Create data
  std::vector<int> in(m * n, 1);
  std::vector<int> out(m * n, 0);
  std::vector<int> res(m * n, 1);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(out, res);
}

TEST(laganina_e_component_labeling_stl, all_one_large) {
  int m = 100;
  int n = 1000;
  // Create data
  std::vector<int> in(m * n, 1);
  std::vector<int> out(m * n, 0);
  std::vector<int> res(m * n, 1);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(out, res);
}

TEST(laganina_e_component_labeling_stl, all_zero) {
  int m = 3;
  int n = 2;
  // Create data
  std::vector<int> in(m * n, 0);
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(in, out);
}

TEST(laganina_e_component_labeling_stl, test1) {
  int m = 3;
  int n = 3;
  // Create data
  std::vector<int> in = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> exp_out = {1, 0, 2, 0, 3, 0, 4, 0, 5};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(exp_out, out);
}

TEST(laganina_e_component_labeling_stl, test2) {
  int m = 4;
  int n = 5;
  // Create data
  std::vector<int> in = {1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1};
  std::vector<int> exp_out = {1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(exp_out, out);
}

TEST(laganina_e_component_labeling_stl, test3) {
  int m = 4;
  int n = 5;
  // Create data
  std::vector<int> in = {1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<int> exp_out = {1, 1, 0, 0, 2, 0, 1, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(exp_out, out);
}

TEST(laganina_e_component_labeling_stl, test6) {
  int m = 4;
  int n = 5;
  // Create data
  std::vector<int> in = {1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<int> exp_out = {1, 1, 0, 0, 2, 0, 1, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(exp_out, out);
}

TEST(laganina_e_component_labeling_stl, test7) {
  int m = 4;
  int n = 5;
  // Create data
  std::vector<int> in = {1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1};
  std::vector<int> exp_out = {1, 1, 0, 0, 2, 1, 1, 1, 0, 2, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(exp_out, out);
}

TEST(laganina_e_component_labeling_stl, one_row) {
  int m = 1;
  int n = 6;
  // Create data
  std::vector<int> in = {1, 0, 1, 1, 0, 1};
  std::vector<int> exp_out = {1, 0, 2, 2, 0, 3};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(exp_out, out);
}

TEST(laganina_e_component_labeling_stl, test4) {
  int m = 4;
  int n = 5;
  // Create data
  std::vector<int> in = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<int> exp_out = {1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(exp_out, out);
}

TEST(laganina_e_component_labeling_stl, test5) {
  int m = 3;
  int n = 3;
  // Create data
  std::vector<int> in = {1, 1, 1, 1, 0, 1, 1, 1, 1};
  std::vector<int> exp_out = {1, 1, 1, 1, 0, 1, 1, 1, 1};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(exp_out, out);
}

TEST(laganina_e_component_labeling_stl, all_one_100) {
  int m = 100;
  int n = 100;
  // Create data
  std::vector<int> in(m * n, 1);
  std::vector<int> out(m * n, 0);
  std::vector<int> exp_out(m * n, 1);

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(exp_out, out);
}

TEST(laganina_e_component_labeling_stl, all_one_300) {
  int m = 300;
  int n = 300;
  // Create data
  std::vector<int> in(m * n, 1);
  std::vector<int> out(m * n, 0);
  std::vector<int> exp_out(m * n, 1);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(exp_out, out);
}

TEST(laganina_e_component_labeling_stl, simple_rectangles_100) {
  int m = 100;
  int n = 100;
  // Create data
  std::vector<int> in(m * n, 0);
  std::vector<int> out(m * n, 0);
  std::vector<int> exp_out(m * n, 0);
  for (int i = 10; i < 40; ++i) {
    for (int j = 20; j < 60; ++j) {
      in[(i * n) + j] = 1;
    }
  }

  for (int i = 60; i < 90; ++i) {
    for (int j = 50; j < 80; ++j) {
      in[(i * n) + j] = 1;
    }
  }

  for (int i = 10; i < 40; ++i) {
    for (int j = 20; j < 60; ++j) {
      exp_out[(i * n) + j] = 1;
    }
  }

  for (int i = 60; i < 90; ++i) {
    for (int j = 50; j < 80; ++j) {
      exp_out[(i * n) + j] = 2;
    }
  }

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(exp_out, out);
}

TEST(laganina_e_component_labeling_stl, diagonal_line_100) {
  int m = 100;
  int n = 100;
  // Create data
  std::vector<int> in(m * n, 0);
  std::vector<int> out(m * n, 0);
  std::vector<int> exp_out(m * n, 0);
  for (int i = 0; i < n; i += 2) {
    in[(i * n) + i] = 1;
  }
  for (int i = 0; i < n; i += 2) {
    exp_out[(i * n) + i] = 1 + (i / 2);
  }

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(exp_out, out);
}

TEST(laganina_e_component_labeling_stl, u_shaped_shape_100) {
  int m = 100;
  int n = 100;
  // Create data data
  std::vector<int> in(m * n, 0);
  std::vector<int> out(m * n, 0);
  std::vector<int> exp_out(m * n, 0);
  for (int i = 10; i < 90; ++i) {
    for (int j = 10; j < 90; ++j) {
      in[(i * n) + j] = 1;
    }
  }
  for (int i = 40; i < 60; ++i) {
    for (int j = 40; j < 60; ++j) {
      in[(i * n) + j] = 0;
    }
  }

  for (int i = 10; i < 90; ++i) {
    for (int j = 10; j < 90; ++j) {
      exp_out[(i * n) + j] = 1;
    }
  }

  for (int i = 40; i < 60; ++i) {
    for (int j = 40; j < 60; ++j) {
      exp_out[(i * n) + j] = 0;
    }
  }
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(exp_out, out);
}

TEST(laganina_e_component_labeling_stl, spiral_pattern) {
  int m = 10;
  int n = 10;
  std::vector<int> in = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1,
                         1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(in, out);
}

TEST(laganina_e_component_labeling_stl, border_pixels) {
  int m = 5;
  int n = 5;

  std::vector<int> in = {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1};

  std::vector<int> exp_out = {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1};

  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(exp_out, out);
}

TEST(laganina_e_component_labeling_stl, ring_with_a_hole_100) {
  int m = 100;
  int n = 100;
  // Create data
  std::vector<int> in(m * n, 0);
  std::vector<int> out(m * n, 0);
  std::vector<int> exp_out(m * n, 0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i < 10 || i >= 90 || j < 10 || j >= 90) {
        in[(i * n) + j] = 1;
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i < 10 || i >= 90 || j < 10 || j >= 90) {
        exp_out[(i * n) + j] = 1;
      }
    }
  }
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(m);
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(m);
  task_data_omp->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_stl::TestTaskSTL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  laganina_e_component_labeling_stl::NormalizeLabels(out);
  EXPECT_EQ(exp_out, out);
}