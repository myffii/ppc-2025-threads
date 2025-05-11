#include "stl/sotskov_a_shell_sorting_with_simple_merging/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

void sotskov_a_shell_sorting_with_simple_merging_stl::ShellSort(std::vector<int>& arr, int left, int right) {
  int array_size = right - left + 1;
  int gap = 1;
  while (gap < array_size / 3) {
    gap = gap * 3 + 1;
  }

  while (gap > 0) {
    for (int i = left + gap; i <= right; ++i) {
      int current_element = arr[i];
      int j = i;
      while (j >= left + gap && arr[j - gap] > current_element) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = current_element;
    }
    gap /= 3;
  }
}

void sotskov_a_shell_sorting_with_simple_merging_stl::ParallelMerge(std::vector<int>& arr, int left, int mid,
                                                                    int right) {
  std::vector<int> temp(right - left + 1);
  int i = left;
  int j = mid + 1;
  int k = 0;

  while (i <= mid && j <= right) {
    temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];
  }
  while (i <= mid) {
    temp[k++] = arr[i++];
  }
  while (j <= right) {
    temp[k++] = arr[j++];
  }
  std::ranges::copy(temp, arr.begin() + left);
}

void sotskov_a_shell_sorting_with_simple_merging_stl::ShellSortWithSimpleMerging(std::vector<int>& arr) {
  int array_size = static_cast<int>(arr.size());
  int num_threads = ppc::util::GetPPCNumThreads();
  int chunk_size = std::max(1, (array_size + num_threads - 1) / num_threads);

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int i = 0; i < num_threads; ++i) {
    int left = i * chunk_size;
    int right = std::min(left + chunk_size - 1, array_size - 1);
    if (left < right) {
      threads.emplace_back(ShellSort, std::ref(arr), left, right);
    }
  }

  std::ranges::for_each(threads, [](auto& thread) { thread.join(); });

  for (int size = chunk_size; size < array_size; size *= 2) {
    std::vector<std::thread> merge_threads;
    for (int i = 0; i < array_size; i += 2 * size) {
      int left = i;
      int mid = std::min(i + size - 1, array_size - 1);
      int right = std::min(i + (2 * size) - 1, array_size - 1);
      if (mid < right) {
        merge_threads.emplace_back(ParallelMerge, std::ref(arr), left, mid, right);
      }
    }

    std::ranges::for_each(merge_threads, [](auto& thread) { thread.join(); });
  }
}

bool sotskov_a_shell_sorting_with_simple_merging_stl::TestTaskSTL::PreProcessingImpl() {
  input_ = std::vector<int>(task_data->inputs_count[0]);
  auto* temp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(temp_ptr, temp_ptr + task_data->inputs_count[0], input_.begin());
  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_stl::TestTaskSTL::ValidationImpl() {
  std::size_t input_size = task_data->inputs_count[0];
  std::size_t output_size = task_data->outputs_count[0];
  if (input_size != output_size) {
    return false;
  }

  for (std::size_t i = 1; i < output_size; ++i) {
    if (task_data->outputs[0][i] < task_data->outputs[0][i - 1]) {
      return false;
    }
  }
  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_stl::TestTaskSTL::RunImpl() {
  ShellSortWithSimpleMerging(input_);
  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_stl::TestTaskSTL::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(input_.begin(), input_.end(), output);
  return true;
}