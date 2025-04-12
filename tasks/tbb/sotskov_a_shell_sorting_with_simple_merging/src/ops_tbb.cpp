#include "tbb/sotskov_a_shell_sorting_with_simple_merging/include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "core/util/include/util.hpp"

void sotskov_a_shell_sorting_with_simple_merging_tbb::ShellSort(std::vector<int>& arr, int left, int right) {
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

void sotskov_a_shell_sorting_with_simple_merging_tbb::ParallelMerge(std::vector<int>& arr, int left, int mid,
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

void sotskov_a_shell_sorting_with_simple_merging_tbb::ShellSortWithSimpleMerging(std::vector<int>& arr) {
  int array_size = static_cast<int>(arr.size());
  int num_threads = ppc::util::GetPPCNumThreads();
  int chunk_size = std::max(1, (array_size + num_threads - 1) / num_threads);

  oneapi::tbb::task_arena arena(num_threads);
  arena.execute([&] {
    oneapi::tbb::parallel_for(0, num_threads, [&](int thread_index) {
      int left = thread_index * chunk_size;
      int right = std::min(left + chunk_size - 1, array_size - 1);
      if (left < right) {
        ShellSort(arr, left, right);
      }
    });

    for (int size = chunk_size; size < array_size; size *= 2) {
      oneapi::tbb::parallel_for(0, (array_size + 2 * size - 1) / (2 * size), [&](int i) {
        int left = i * 2 * size;
        int mid = std::min(left + size - 1, array_size - 1);
        int right = std::min(left + (2 * size - 1), array_size - 1);
        if (mid < right) {
          ParallelMerge(arr, left, mid, right);
        }
      });
    }
  });
}

bool sotskov_a_shell_sorting_with_simple_merging_tbb::TestTaskTBB::PreProcessingImpl() {
  input_ = std::vector<int>(task_data->inputs_count[0]);
  auto* temp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(temp_ptr, temp_ptr + task_data->inputs_count[0], input_.begin());
  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_tbb::TestTaskTBB::ValidationImpl() {
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

bool sotskov_a_shell_sorting_with_simple_merging_tbb::TestTaskTBB::RunImpl() {
  int num_threads = ppc::util::GetPPCNumThreads();
  oneapi::tbb::task_arena arena(num_threads);
  arena.execute([&] { ShellSortWithSimpleMerging(input_); });
  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_tbb::TestTaskTBB::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(input_.begin(), input_.end(), output);
  return true;
}
