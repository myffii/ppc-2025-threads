#include "omp/shlyakov_m_shell_sort_omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <vector>

bool shlyakov_m_shell_sort_omp::TestTaskOpenMP::PreProcessingImpl() {
  std::size_t input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  output_ = input_;

  return true;
}

bool shlyakov_m_shell_sort_omp::TestTaskOpenMP::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool shlyakov_m_shell_sort_omp::TestTaskOpenMP::RunImpl() {
  int array_size = static_cast<int>(input_.size());
  int num_threads = omp_get_max_threads();
  int sub_arr_size = (array_size + num_threads - 1) / num_threads;

#pragma omp parallel
  {
    std::vector<int> buffer;

#pragma omp for schedule(dynamic)
    for (int i = 0; i < num_threads; ++i) {
      int left = i * sub_arr_size;
      int right = (left + sub_arr_size - 1 < array_size - 1) ? (left + sub_arr_size - 1) : (array_size - 1);

      if (left < right) {
        ShellSort(left, right, input_);
      }
    }

    for (int size = sub_arr_size; size < array_size; size *= 2) {
#pragma omp for schedule(dynamic)
      for (int left = 0; left < array_size; left += 2 * size) {
        int mid = (left + size - 1 < array_size - 1) ? (left + size - 1) : (array_size - 1);
        int right_bound = left + (2 * size) - 1;
        int right = (right_bound < array_size - 1) ? right_bound : (array_size - 1);

        if (mid < right) {
          Merge(left, mid, right, input_, buffer);
        }
      }
    }
  }

  output_ = input_;
  return true;
}

namespace shlyakov_m_shell_sort_omp {
void ShellSort(int left, int right, std::vector<int>& arr) {
  int sub_array_size = right - left + 1;
  int gap = 1;

  for (; gap <= sub_array_size / 3;) {
    gap = gap * 3 + 1;
  }

  for (; gap > 0; gap /= 3) {
    for (int k = left + gap; k <= right; ++k) {
      int current_element = arr[k];
      int j = k;

      while (j >= left + gap && arr[j - gap] > current_element) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = current_element;
    }
  }
}

void Merge(int left, int mid, int right, std::vector<int>& arr, std::vector<int>& buffer) {
  int i = left;
  int j = mid + 1;
  int k = 0;
  const int merge_size = right - left + 1;

  if (buffer.size() < static_cast<std::size_t>(merge_size)) {
    buffer.resize(static_cast<std::size_t>(merge_size));
  }

  for (; i <= mid || j <= right;) {
    if (i > mid) {
      buffer[k++] = arr[j++];
    } else if (j > right) {
      buffer[k++] = arr[i++];
    } else {
      buffer[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    }
  }

  for (size_t idx = 0; idx < static_cast<size_t>(k); ++idx) {
    arr[left + idx] = buffer[idx];
  }
}
}  // namespace shlyakov_m_shell_sort_omp

bool shlyakov_m_shell_sort_omp::TestTaskOpenMP::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
// namespace shlyakov_m_shell_sort_omp