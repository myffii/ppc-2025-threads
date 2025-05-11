#include "stl/ermilova_d_shell_sort_batcher_even_odd_merger/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace {

std::vector<int> CreateSedgwickSequence(int n) {
  std::vector<int> gaps;
  int k = 0;
  while (true) {
    int gap =
        (k % 2 == 0) ? (9 * (1 << (2 * k))) - (9 * (1 << k)) + 1 : (8 * (1 << k)) - (6 * (1 << ((k + 1) / 2))) + 1;

    if (gap > n / 2) {
      break;
    }

    gaps.push_back(gap);
    k++;
  }

  if (gaps.empty() || gaps.back() != 1) {
    gaps.push_back(1);
  }

  std::ranges::reverse(gaps);
  return gaps;
}

void ShellSort(std::vector<int> &data, size_t start, size_t end) {
  auto partition_size = static_cast<int>(end - start + 1);
  auto gaps = CreateSedgwickSequence(partition_size);

  for (int gap : gaps) {
    for (size_t i = start + gap; i <= end; i++) {
      int temp = data[i];
      size_t j = i;
      while (j >= start + gap && data[j - gap] > temp) {
        data[j] = data[j - gap];
        j -= gap;
      }
      data[j] = temp;
    }
  }
}

void BatcherMerge(std::vector<int> &data, size_t start, size_t mid, size_t end) {
  std::vector<int> left(data.begin() + static_cast<std::ptrdiff_t>(start),
                        data.begin() + static_cast<std::ptrdiff_t>(mid));

  std::vector<int> right(data.begin() + static_cast<std::ptrdiff_t>(mid),
                         data.begin() + static_cast<std::ptrdiff_t>(end));
  size_t left_index = 0;
  size_t right_index = 0;
  size_t data_offset = start;

  size_t left_size = mid - start;
  size_t right_size = end - mid;

  for (size_t i = start; i < end; ++i) {
    if (i % 2 == 0) {
      if (left_index < left_size && (right_index >= right_size || left[left_index] <= right[right_index])) {
        data[data_offset++] = left[left_index++];
      } else {
        data[data_offset++] = right[right_index++];
      }
    } else {
      if (right_index < right_size && (left_index >= left_size || right[right_index] <= left[left_index])) {
        data[data_offset++] = right[right_index++];
      } else {
        data[data_offset++] = left[left_index++];
      }
    }
  }
}

void ParallelShellSortWithBatcherMerge(std::vector<int> &data) {
  size_t elements_count = data.size();
  if (elements_count <= 1) {
    return;
  }

  int threads_count = ppc::util::GetPPCNumThreads();
  size_t block_size = (elements_count + threads_count - 1) / threads_count;
  std::vector<std::thread> threads(threads_count);
  for (int i = 0; i < threads_count; i++) {
    size_t start_block_index = static_cast<size_t>(i) * block_size;
    size_t end_block_index = std::min(start_block_index + block_size, elements_count) - 1;
    if (start_block_index < elements_count) {
      threads[i] = std::thread(ShellSort, std::ref(data), start_block_index, end_block_index);
    }
  }

  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  for (size_t merge_size = block_size; merge_size < elements_count; merge_size *= 2) {
    size_t num_merges = (elements_count + 2 * merge_size - 1) / (2 * merge_size);
    std::vector<std::thread> merge_threads(num_merges);

    for (size_t i = 0; i < num_merges; ++i) {
      size_t left = i * 2 * merge_size;
      size_t mid = std::min(left + merge_size, elements_count);
      size_t right = std::min(left + (2 * merge_size), elements_count);

      if (mid < right) {
        merge_threads[i] = std::thread(BatcherMerge, std::ref(data), left, mid, right);
      }
    }

    for (auto &t : merge_threads) {
      if (t.joinable()) {
        t.join();
      }
    }
  }
}
}  // namespace
bool ermilova_d_shell_sort_batcher_even_odd_merger_stl::StlTask::PreProcessingImpl() {
  auto input_task_size = task_data->inputs_count[0];
  auto *input_task_data = reinterpret_cast<int *>(task_data->inputs[0]);
  data_ = std::vector(input_task_data, input_task_data + input_task_size);

  return true;
}

bool ermilova_d_shell_sort_batcher_even_odd_merger_stl::StlTask::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool ermilova_d_shell_sort_batcher_even_odd_merger_stl::StlTask::RunImpl() {
  ParallelShellSortWithBatcherMerge(data_);
  return true;
}

bool ermilova_d_shell_sort_batcher_even_odd_merger_stl::StlTask::PostProcessingImpl() {
  auto *output_task_data = reinterpret_cast<int *>(task_data->outputs[0]);
  std::ranges::copy(data_, output_task_data);
  return true;
}
