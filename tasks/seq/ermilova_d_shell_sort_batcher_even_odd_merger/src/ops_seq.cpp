#include "seq/ermilova_d_shell_sort_batcher_even_odd_merger/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace {

std::vector<size_t> CreateSedgwickSequence(int n) {
  std::vector<size_t> gaps;
  int k = 0;
  while (true) {
    int gap =
        (k % 2 == 0) ? (9 * (1 << (2 * k))) - (9 * (1 << k)) + 1 : (8 * (1 << k)) - (6 * (1 << ((k + 1) / 2))) + 1;

    if (gap > n / 2) {
      break;
    }

    gaps.push_back(static_cast<size_t>(gap));
    k++;
  }

  if (gaps.empty() || gaps.back() != 1) {
    gaps.push_back(1);
  }

  std::ranges::reverse(gaps);
  return gaps;
}

void ShellSort(std::vector<int> &data) {
  auto elements_count = data.size();
  auto gaps = CreateSedgwickSequence(static_cast<int>(elements_count));

  for (size_t gap : gaps) {
    for (size_t i = gap; i < elements_count; i++) {
      int temp = data[i];
      size_t j = i;
      while (j >= gap && data[j - gap] > temp) {
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

  ShellSort(data);

  size_t mid = elements_count / 2;
  size_t end = elements_count;
  if (mid < end) {
    BatcherMerge(data, 0, mid, end);
  }
}
}  // namespace

bool ermilova_d_shell_sort_batcher_even_odd_merger_seq::SequentialTask::PreProcessingImpl() {
  auto input_task_size = task_data->inputs_count[0];
  auto *input_task_data = reinterpret_cast<int *>(task_data->inputs[0]);
  data_ = std::vector(input_task_data, input_task_data + input_task_size);

  return true;
}

bool ermilova_d_shell_sort_batcher_even_odd_merger_seq::SequentialTask::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool ermilova_d_shell_sort_batcher_even_odd_merger_seq::SequentialTask::RunImpl() {
  ParallelShellSortWithBatcherMerge(data_);
  return true;
}

bool ermilova_d_shell_sort_batcher_even_odd_merger_seq::SequentialTask::PostProcessingImpl() {
  auto *output_task_data = reinterpret_cast<int *>(task_data->outputs[0]);
  std::ranges::copy(data_, output_task_data);
  return true;
}
