#include <algorithm>
#include <cmath>
#include <cstddef>     // Для std::size_t
#include <functional>  // Для std::plus и std::minus
#include <future>
#include <thread>
#include <utility>  // Для std::move
#include <vector>

#include "stl/nasedkin_e_strassen_algorithm/include/ops_stl.hpp"

namespace nasedkin_e_strassen_algorithm_stl {

bool StrassenStl::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto *terno, in_ptr_b = reinterpret_cast<double*>(task_data->inputs[1]);

  matrix_size_ = static_cast<int>(std::sqrt(input_size));
  input_matrix_a_.resize(matrix_size_ * matrix_size_);
  input_matrix_b_.resize(matrix_size_ * matrix_size_);

  std::copy(in_ptr_a, in_ptr_a + input_size, input_matrix_a_.begin());
  std::copy(in_ptr_b, in_ptr_b + input_size, input_matrix_b_.begin());

  if ((matrix_size_ & (matrix_size_ - 1)) != 0) {
    original_size_ = matrix_size_;
    input_matrix_a_ = PadMatrixToPowerOfTwo(input_matrix_a_, matrix_size_);
    input_matrix_b_ = PadMatrixToPowerOfTwo(input_matrix_b_, matrix_size_);
    matrix_size_ = static_cast<int>(std::sqrt(input_matrix_a_.size()));
  } else {
    original_size_ = matrix_size_;
  }

  output_matrix_.resize(matrix_size_ * matrix_size_, 0.0);
  return true;
}

bool StrassenStl::ValidationImpl() {
  unsigned int input_size_a = task_data->inputs_count[0];
  unsigned int input_size_b = task_data->inputs_count[1];
  unsigned int output_size = task_data->outputs_count[0];

  if (input_size_a == 0 || input_size_b == 0 || output_size == 0) {
    return false;
  }

  int size_a = static_cast<int>(std::sqrt(input_size_a));
  int size_b = static_cast<int>(std::sqrt(input_size_b));
  int size_output = static_cast<int>(std::sqrt(output_size));

  return (size_a == size_b) && (size_a == size_output);
}

bool StrassenStl::RunImpl() {
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_);
  return true;
}

bool StrassenStl::PostProcessingImpl() {
  if (original_size_ != matrix_size_) {
    output_matrix_ = TrimMatrixToOriginalSize(output_matrix_, original_size_, matrix_size_);
  }

  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(output_matrix_, out_ptr);
  return true;
}

// Обновлено в соответствии с объявлением в ops_stl.hpp
std::vector<double> StrassenStl::AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::plus<>());
  return result;
}

// Обновлено в соответствии с объявлением в ops_stl.hpp
std::vector<double> StrassenStl::SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b,
                                                  int size) {
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::minus<>());
  return result;
}

std::vector<double> StandardMultiply(const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::vector<double> result(size * size, 0.0);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        result[(i * size) + j] += a[(i * size) + k] * b[(k * size) + j];
      }
    }
  }
  return result;
}

std::vector<double> StrassenStl::PadMatrixToPowerOfTwo(const std::vector<double>& matrix, int original_size) {
  int new_size = 1;
  while (new_size < original_size) {
    new_size *= 2;
  }

  std::vector<double> padded_matrix(new_size * new_size, 0);
  for (int i = 0; i < original_size; ++i) {
    std::copy(matrix.begin() + i * original_size, matrix.begin() + (i + 1) * original_size,
              padded_matrix.begin() + i * new_size);
  }
  return padded_matrix;
}

std::vector<double> StrassenStl::TrimMatrixToOriginalSize(const std::vector<double>& matrix, int original_size,
                                                          int padded_size) {
  std::vector<double> trimmed_matrix(original_size * original_size);
  for (int i = 0; i < original_size; ++i) {
    std::copy(matrix.begin() + i * padded_size, matrix.begin() + i * padded_size + original_size,
              trimmed_matrix.begin() + i * original_size);
  }
  return trimmed_matrix;
}

std::vector<double> StrassenStl::StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b,
                                                  int size) {
  if (size <= 64) {  // Порог остаётся 64
    return StandardMultiply(a, b, size);
  }

  int half_size = size / 2;
  int half_size_squared = half_size * half_size;

  std::vector<double> a11(half_size_squared);
  std::vector<double> a12(half_size_squared);
  std::vector<double> a21(half_size_squared);
  std::vector<double> a22(half_size_squared);
  std::vector<double> b11(half_size_squared);
  std::vector<double> b12(half_size_squared);
  std::vector<double> b21(half_size_squared);
  std::vector<double> b22(half_size_squared);

  SplitMatrix(a, a11, 0, 0, size);
  SplitMatrix(a, a12, 0, half_size, size);
  SplitMatrix(a, a21, half_size, 0, size);
  SplitMatrix(a, a22, half_size, half_size, size);

  SplitMatrix(b, b11, 0, 0, size);
  SplitMatrix(b, b12, 0, half_size, size);
  SplitMatrix(b, b21, half_size, 0, size);
  SplitMatrix(b, b22, half_size, half_size, size);

  // Определяем количество доступных ядер
  static unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 2;  // Если не удалось определить, предполагаем 2 ядра
  }

  // Параллельное выполнение с использованием std::thread и std::promise/std::future
  const auto compute_task = [&](std::size_t task_id, std::promise<std::vector<double>>&& promise) {
    std::vector<double> local_temp(half_size_squared);
    std::vector<double> local_temp2(half_size_squared);
    std::vector<double> result(half_size_squared);
    switch (task_id) {
      case 0:  // p1 = (a11 + a22) * (b11 + b22)
        local_temp = AddMatrices(a11, a22, half_size);
        local_temp2 = AddMatrices(b11, b22, half_size);
        result = StrassenMultiply(local_temp, local_temp2, half_size);
        break;
      case 1:  // p2 = (a21 + a22) * b11
        local_temp = AddMatrices(a21, a22, half_size);
        result = StrassenMultiplyward, StrassenMultiply(local_temp, b11, half_size);
        break;
      case 2:  // p3 = a11 * (b12 - b22)
        local_temp = SubtractMatrices(b12, b22, half_size);
        result = StrassenMultiply(a11, local_temp, half_size);
        break;
      case 3:  // p4 = a22 * (b21 - b11)
        local_temp = SubtractMatrices(b21, b11, half_size);
        result = StrassenMultiply(a22, local_temp, half_size);
        break;
      case 4:  // p5 = (a11 + a12) * b22
        local_temp = AddMatrices(a11, a12, half_size);
        result = StrassenMultiply(local_temp, b22, half_size);
        break;
      case 5:  // p6 = (a21 - a11) * (b11 + b12)
        local_temp = SubtractMatrices(a21, a11, half_size);
        local_temp2 = AddMatrices(b11, b12, half_size);
        result = StrassenMultiply(local_temp, local_temp2, half_size);
        break;
      case 6:  // p7 = (a12 - a22) * (b21 + b22)
        local_temp = SubtractMatrices(a12, a22, half_size);
        local_temp2 = AddMatrices(b21, b22, half_size);
        result = StrassenMultiply(local_temp, local_temp2, half_size);
        break;
      default:
        break;  // Добавлен default для switch
    }
    promise.set_value(result);
  };

  const std::size_t num_tasks = 7;  // Количество задач (p1–p7)
  std::vector<std::future<std::vector<double>>> futures(num_tasks);
  std::vector<std::thread> threads(std::min(num_tasks, static_cast<std::size_t>(num_threads)));

  // Распределяем задачи по потокам
  for (std::size_t i = 0; i < num_tasks; i += threads.size()) {
    std::size_t current_batch_size = std::min(threads.size(), num_tasks - i);
    for (std::size_t j = 0; j < current_batch_size; ++j) {
      std::promise<std::vector<double>> promise;
      futures[i + j] = promise.get_future();
      threads[j] = std::thread(compute_task, i + j, std::move(promise));
    }
    for (std::size_t j = 0; j < current_batch_size; ++j) {
      threads[j].join();
    }
  }

  // Собираем результаты
  std::vector<double> p1 = futures[0].get();
  std::vector<double> p2 = futures[1].get();
  std::vector<double> p3 = futures[2].get();
  std::vector<double> p4 = futures[3].get();
  std::vector<double> p5 = futures[4].get();
  std::vector<double> p6 = futures[5].get();
  std::vector<double> p7 = futures[6].get();

  // Вычисляем результирующие подматрицы
  std::vector<double> c11(half_size_squared);
  std::vector<double> c12(half_size_squared);
  std::vector<double> c21(half_size_squared);
  std::vector<double> c22(half_size_squared);

  // c11 = p1 + p4 - p5 + p7
  auto temp = AddMatrices(p1, p4, half_size);
  auto temp2 = SubtractMatrices(temp, p5, half_size);
  c11 = AddMatrices(temp2, p7, half_size);

  // c12 = p3 + p5
  c12 = AddMatrices(p3, p5, half_size);

  // c21 = p2 + p4
  c21 = AddMatrices(p2, p4, half_size);

  // c22 = p1 + p3 - p2 + p6
  temp = AddMatrices(p1, p3, half_size);
  temp2 = SubtractMatrices(temp, p2, half_size);
  c22 = AddMatrices(temp2, p6, half_size);

  std::vector<double> result(size * size);
  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);

  return result;
}

void StrassenStl::SplitMatrix(const std::vector<double>& parent, std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::copy(parent.begin() + ((row_start + i) * parent_size + col_start),
              parent.begin() + ((row_start + i) * parent_size + col_start + child_size),
              child.begin() + (i * child_size));
  }
}

void StrassenStl::MergeMatrix(std::vector<double>& parent, const std::vector<double>& child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::copy(child.begin() + (i * child_size), child.begin() + ((i + 1) * child_size),
              parent.begin() + ((row_start + i) * parent_size + col_start));
  }
}

}  // namespace nasedkin_e_strassen_algorithm_stl