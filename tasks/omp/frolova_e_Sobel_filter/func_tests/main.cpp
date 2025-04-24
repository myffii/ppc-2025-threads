#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/frolova_e_Sobel_filter/include/ops_omp.hpp"

namespace {
std::vector<int> GenRgbPicture(size_t width, size_t height) {
  std::vector<int> image(width * height * 3);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> rgb(0, 255);

  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      size_t index = (y * width + x) * 3;
      image[index] = rgb(gen);
      image[index + 1] = rgb(gen);
      image[index + 2] = rgb(gen);
    }
  }

  return image;
}

std::vector<frolova_e_sobel_filter_omp::RGB> ConvertToRGB(const std::vector<int> &pict) {
  std::vector<frolova_e_sobel_filter_omp::RGB> picture;
  size_t pixel_count = pict.size() / 3;

  for (size_t i = 0; i < pixel_count; i++) {
    frolova_e_sobel_filter_omp::RGB pixel;
    pixel.R = pict[i * 3];
    pixel.G = pict[(i * 3) + 1];
    pixel.B = pict[(i * 3) + 2];

    picture.push_back(pixel);
  }
  return picture;
}

std::vector<int> ToGrayScaleImg(std::vector<frolova_e_sobel_filter_omp::RGB> &color_img, size_t width, size_t height) {
  std::vector<int> gray_scale_image(width * height);

#pragma omp for schedule(static)
  for (int i = 0; i < static_cast<int>(width * height); i++) {
    gray_scale_image[i] =
        static_cast<int>((0.299 * color_img[i].R) + (0.587 * color_img[i].G) + (0.114 * color_img[i].B));
  }

  return gray_scale_image;
}

std::vector<int> ApplySobelFilter(const std::vector<int> &gray_scale_image, size_t width, size_t height) {
  const std::vector<int> gx = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  const std::vector<int> gy = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

  std::vector<int> result(width * height, 0);

  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      int res_x = 0;
      int res_y = 0;

      for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
          int px = static_cast<int>(x) + kx;
          int py = static_cast<int>(y) + ky;

          int pixel_value = 0;

          if (px >= 0 && px < static_cast<int>(width) && py >= 0 && py < static_cast<int>(height)) {
            pixel_value = gray_scale_image[(py * width) + px];
          }

          size_t kernel_ind = ((ky + 1) * 3) + (kx + 1);
          res_x += pixel_value * gx[kernel_ind];
          res_y += pixel_value * gy[kernel_ind];
        }
      }
      int gradient = static_cast<int>(std::sqrt((res_x * res_x) + (res_y * res_y)));
      result[(y * width) + x] = std::clamp(gradient, 0, 255);
    }
  }

  return result;
}

}  // namespace

TEST(frolova_e_sobel_filter_omp, small_image_1) {
  std::vector<int> value = {3, 3};
  std::vector<int> pict = {172, 47,  117, 192, 67, 251, 195, 103, 9,  211, 21, 242, 3,  87,
                           70,  216, 88,  140, 58, 193, 230, 39,  87, 174, 88, 81,  165};

  std::vector<int> res(9, 0);

  std::vector<int> reference = {255, 255, 255, 255, 53, 255, 255, 255, 255};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
  task_data->inputs_count.emplace_back(value.size());

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
  task_data->inputs_count.emplace_back(pict.size());

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(res.size());

  // Create Task
  frolova_e_sobel_filter_omp::SobelFilterOmp test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  EXPECT_EQ(reference, res);
}

TEST(frolova_e_sobel_filter_omp, test_1) {
  std::vector<int> value = {10, 10};
  std::vector<int> pict = {
      172, 47,  117, 192, 67,  251, 195, 103, 9,   211, 21,  242, 36,  87,  70,  216, 88,  140, 58,  193, 230, 39,
      87,  174, 88,  81,  165, 25,  77,  72,  9,   148, 115, 208, 243, 197, 254, 79,  175, 192, 82,  99,  216, 177,
      243, 29,  147, 147, 142, 167, 32,  193, 9,   185, 127, 32,  31,  202, 244, 151, 163, 254, 203, 114, 183, 28,
      34,  128, 128, 164, 53,  133, 38,  232, 244, 17,  79,  132, 105, 42,  186, 31,  120, 1,   65,  231, 169, 57,
      35,  102, 119, 11,  174, 82,  91,  128, 142, 99,  53,  140, 121, 170, 84,  203, 68,  6,   196, 47,  127, 244,
      131, 204, 100, 180, 232, 78,  143, 148, 227, 186, 23,  207, 141, 117, 85,  48,  49,  69,  169, 163, 192, 95,
      197, 94,  0,   113, 178, 36,  162, 48,  93,  131, 98,  42,  205, 112, 231, 149, 201, 127, 0,   138, 114, 43,
      186, 127, 23,  187, 130, 121, 98,  62,  163, 222, 123, 195, 82,  174, 227, 148, 209, 50,  155, 14,  41,  58,
      193, 36,  10,  86,  43,  104, 11,  2,   51,  80,  32,  182, 128, 38,  19,  174, 42,  115, 184, 188, 232, 77,
      30,  24,  125, 2,   3,   94,  226, 107, 13,  112, 40,  72,  19,  95,  72,  154, 194, 248, 180, 67,  236, 61,
      14,  96,  4,   195, 237, 139, 252, 86,  205, 121, 109, 75,  184, 16,  152, 157, 149, 110, 25,  208, 188, 121,
      118, 117, 189, 83,  161, 104, 160, 228, 251, 251, 121, 70,  213, 31,  13,  71,  184, 152, 79,  41,  18,  40,
      182, 207, 11,  166, 111, 93,  249, 129, 223, 118, 44,  216, 125, 24,  67,  210, 239, 3,   234, 204, 230, 35,
      214, 254, 189, 197, 215, 43,  32,  11,  104, 212, 138, 182, 235, 165};

  std::vector<int> res(100, 0);

  std::vector<int> reference = {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 146, 255, 151, 138, 155, 244,
                                135, 255, 255, 255, 255, 255, 255, 95,  206, 171, 239, 221, 255, 255, 232, 116, 218,
                                84,  107, 118, 46,  194, 255, 255, 157, 179, 188, 69,  39,  105, 153, 255, 255, 255,
                                129, 70,  255, 205, 132, 255, 255, 246, 255, 255, 209, 183, 189, 255, 153, 255, 134,
                                244, 255, 255, 255, 255, 255, 255, 238, 255, 234, 168, 255, 255, 184, 156, 255, 255,
                                104, 196, 135, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
  task_data->inputs_count.emplace_back(value.size());

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
  task_data->inputs_count.emplace_back(pict.size());

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(res.size());

  // Create Task
  frolova_e_sobel_filter_omp::SobelFilterOmp test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  EXPECT_EQ(reference, res);
}

TEST(frolova_e_sobel_filter_omp, small_image_2) {
  std::vector<int> value = {5, 5};
  std::vector<int> pict = {172, 47,  117, 192, 67,  251, 195, 103, 9,   211, 21,  242, 36,  87,  70,  216, 88,  140, 58,
                           193, 230, 39,  87,  174, 88,  81,  165, 25,  77,  72,  9,   148, 115, 208, 243, 197, 254, 79,
                           175, 192, 82,  99,  216, 177, 243, 29,  147, 147, 142, 167, 32,  193, 9,   185, 127, 32,  31,
                           202, 244, 151, 163, 254, 203, 114, 183, 28,  34,  128, 128, 164, 53,  133, 38,  232, 244};

  std::vector<int> res(25, 0);

  std::vector<int> reference = {255, 255, 255, 255, 255, 255, 239, 255, 180, 255, 255, 43, 255,
                                242, 255, 255, 162, 255, 255, 255, 255, 255, 255, 255, 255};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
  task_data->inputs_count.emplace_back(value.size());

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
  task_data->inputs_count.emplace_back(pict.size());

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(res.size());

  // Create Task
  frolova_e_sobel_filter_omp::SobelFilterOmp test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  EXPECT_EQ(reference, res);
}

TEST(frolova_e_sobel_filter_omp, _1000_1000_picture) {
  std::vector<int> value = {1000, 1000};
  std::vector<int> pict = GenRgbPicture(1000, 1000);

  std::vector<int> res(1000000, 0);
  std::vector<int> reference(1000000, 0);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
  task_data->inputs_count.emplace_back(value.size());

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
  task_data->inputs_count.emplace_back(pict.size());

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(res.size());

  // Create Task
  frolova_e_sobel_filter_omp::SobelFilterOmp test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);

  std::vector<frolova_e_sobel_filter_omp::RGB> picture = ConvertToRGB(pict);
  std::vector<int> gray_scale_image =
      ToGrayScaleImg(picture, static_cast<size_t>(value[0]), static_cast<size_t>(value[1]));

  reference = ApplySobelFilter(gray_scale_image, value[0], value[1]);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  EXPECT_EQ(reference, res);
}

//______ASSERT_FALSE________________

TEST(frolova_e_sobel_filter_omp, not_correct_value) {
  std::vector<int> value = {-1, 1};
  std::vector<int> pict = {100, 0, 0};

  std::vector<int> res(1, 0);
  std::vector<int> reference = {0};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
  task_data->inputs_count.emplace_back(value.size());

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
  task_data->inputs_count.emplace_back(pict.size());

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(res.size());

  // Create Task
  frolova_e_sobel_filter_omp::SobelFilterOmp test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}

TEST(frolova_e_sobel_filter_omp, vector_is_not_multiple_of_three) {
  std::vector<int> value = {1, 1};
  std::vector<int> pict = {100, 0};

  std::vector<int> res(1, 0);

  std::vector<int> reference = {0};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
  task_data->inputs_count.emplace_back(value.size());

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
  task_data->inputs_count.emplace_back(pict.size());

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(res.size());

  // Create Task
  frolova_e_sobel_filter_omp::SobelFilterOmp test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}

TEST(frolova_e_sobel_filter_omp, vector_element_is_not_included_the_range) {
  std::vector<int> value = {1, 1};
  std::vector<int> pict = {100, 0, 270};

  std::vector<int> res(1, 0);

  std::vector<int> reference = {0};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
  task_data->inputs_count.emplace_back(value.size());

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
  task_data->inputs_count.emplace_back(pict.size());

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(res.size());

  // Create Task
  frolova_e_sobel_filter_omp::SobelFilterOmp test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}

TEST(frolova_e_sobel_filter_omp, negative_value_of_element_int_RGBvector) {
  std::vector<int> value = {1, 1};
  std::vector<int> pict = {100, 0, -1};

  std::vector<int> res(1, 0);

  std::vector<int> reference = {0};

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
  task_data->inputs_count.emplace_back(value.size());

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
  task_data->inputs_count.emplace_back(pict.size());

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(res.size());

  // Create Task
  frolova_e_sobel_filter_omp::SobelFilterOmp test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}