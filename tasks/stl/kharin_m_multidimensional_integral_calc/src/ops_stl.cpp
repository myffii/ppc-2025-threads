#include "stl/kharin_m_multidimensional_integral_calc/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

bool kharin_m_multidimensional_integral_calc_stl::TaskSTL::ValidationImpl() {
  // Проверяем, что предоставлено ровно 3 входа и 1 выход
  if (task_data->inputs.size() != 3 || task_data->outputs.size() != 1) {
    return false;
  }
  // Совпадение grid_sizes и step_sizes
  if (task_data->inputs_count[1] != task_data->inputs_count[2]) {
    return false;
  }
  // Выход должен содержать одно значение
  if (task_data->outputs_count[0] != 1) {
    return false;
  }
  return true;
}

bool kharin_m_multidimensional_integral_calc_stl::TaskSTL::PreProcessingImpl() {
  size_t d = task_data->inputs_count[1];
  auto* sizes_ptr = reinterpret_cast<size_t*>(task_data->inputs[1]);
  grid_sizes_ = std::vector<size_t>(sizes_ptr, sizes_ptr + d);

  // Вычисляем общее количество точек сетки
  size_t total_size = 1;
  for (const auto& n : grid_sizes_) {
    total_size *= n;
  }

  if (task_data->inputs_count[0] != total_size) {
    return false;
  }
  auto* input_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_ = std::vector<double>(input_ptr, input_ptr + total_size);

  if (task_data->inputs_count[2] != d) {
    return false;
  }
  auto* steps_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
  step_sizes_ = std::vector<double>(steps_ptr, steps_ptr + d);

  // Проверка на отрицательные шаги
  for (const auto& h : step_sizes_) {
    if (h <= 0.0) {
      return false;  // Отрицательный или нулевой шаг недопустим
    }
  }

  output_result_ = 0.0;
  return true;
}

bool kharin_m_multidimensional_integral_calc_stl::TaskSTL::RunImpl() {
  if (input_.empty()) {
    output_result_ = 0.0;
    return true;  // Интеграл от пустого множества - 0
  }

  num_threads_ = std::min(static_cast<size_t>(ppc::util::GetPPCNumThreads()), input_.size());
  std::vector<std::thread> threads;
  threads.reserve(num_threads_);
  std::vector<double> partial_sums(num_threads_, 0);

  auto input_chunk_size = input_.size() / num_threads_;
  auto remainder = input_.size() % num_threads_;

  auto chunk_plus = [&](std::vector<double>::iterator it_begin, size_t size, double& result_location) {
    double local = 0;
    for (size_t i = 0; i < size; ++i) {
      // Cast size_t to iterator difference_type to avoid narrowing conversion
      local += *(it_begin + static_cast<std::vector<double>::difference_type>(i));
    }
    result_location = local;
  };

  size_t current_start_index = 0;
  for (size_t i = 0; i < num_threads_; ++i) {
    size_t size = (i < remainder) ? (input_chunk_size + 1) : input_chunk_size;
    auto it_begin = input_.begin() + static_cast<std::vector<double>::difference_type>(current_start_index);
    std::thread th(chunk_plus, it_begin, size, std::ref(partial_sums[i]));
    threads.push_back(std::move(th));
    current_start_index += size;
  }

  for (auto& th : threads) {
    if (th.joinable()) {
      //  th.join();
      th.join();
    }
  }

  double total = 0;
  for (const auto& partial : partial_sums) {
    total += partial;
  }

  // Вычисляем элемент объема как произведение шагов интегрирования
  double volume_element = 1.0;
  for (const auto& h : step_sizes_) {
    volume_element *= h;
  }
  // Интеграл = сумма значений * элемент объема
  output_result_ = total * volume_element;
  return true;
}

bool kharin_m_multidimensional_integral_calc_stl::TaskSTL::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = output_result_;
  return true;
}