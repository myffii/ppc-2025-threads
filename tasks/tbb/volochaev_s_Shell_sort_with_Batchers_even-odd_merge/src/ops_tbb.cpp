#include "tbb/volochaev_s_Shell_sort_with_Batchers_even-odd_merge/include/ops_tbb.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <vector>

#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task_arena.h"

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB::PreProcessingImpl() {
  // Init value for input and output
  size_ = static_cast<int>(task_data->inputs_count[0]);

  auto* input_pointer = reinterpret_cast<int*>(task_data->inputs[0]);
  array_ = std::vector<int>(input_pointer, input_pointer + size_);
  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB::ShellSort(int start) {
  int n = mini_batch_;

  int gap = 1;
  while (gap < n / 3) {
    gap = 3 * gap + 1;
  }

  while (gap >= 1) {
    for (int i = start + gap; i < start + mini_batch_; ++i) {
      int temp = mass_[i];
      int j = i;
      while (j >= start + gap && mass_[j - gap] > temp) {
        mass_[j] = mass_[j - gap];
        j -= gap;
      }
      mass_[j] = temp;
    }
    gap /= 3;
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB::MergeBlocks(int id_l, int id_r, int len) {
  int left_id = 0;
  int right_id = 0;
  int merged_id = 0;

  while (left_id < len || right_id < len) {
    if (left_id < len && right_id < len) {
      if (mass_[id_l + left_id] < mass_[id_r + right_id]) {
        array_[id_l + merged_id] = mass_[id_l + left_id];
        left_id += 2;
      } else {
        array_[id_l + merged_id] = mass_[id_r + right_id];
        right_id += 2;
      }
    } else if (left_id < len) {
      array_[id_l + merged_id] = mass_[id_l + left_id];
      left_id += 2;
    } else {
      array_[id_l + merged_id] = mass_[id_r + right_id];
      right_id += 2;
    }
    merged_id += 2;
  }

  for (int i = 0; i < merged_id; i += 2) {
    mass_[id_l + i] = array_[id_l + i];
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB::LastMerge() {
  int even_index = 0;
  int odd_index = 1;
  int result_index = 0;

  while (even_index < n_ || odd_index < n_) {
    if (even_index < n_ && odd_index < n_) {
      if (mass_[even_index] < mass_[odd_index]) {
        array_[result_index++] = mass_[even_index];
        even_index += 2;
      } else {
        array_[result_index++] = mass_[odd_index];
        odd_index += 2;
      }
    } else if (even_index < n_) {
      array_[result_index++] = mass_[even_index];
      even_index += 2;
    } else {
      array_[result_index++] = mass_[odd_index];
      odd_index += 2;
    }
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB::Merge() {
  tbb::task_arena arena(c_threads_);

  for (int i = c_threads_; i > 1; i /= 2) {
    arena.execute([&] {
      tbb::parallel_for(0, i / 2, [&](int id) {
        int l = mini_batch_ * (c_threads_ / i);
        for (int ost = 0; ost < 2; ++ost) {
          MergeBlocks((id * 2 * l) + ost, (id * 2 * l) + l + ost, l - ost);
        }
      });
    });
  }

  LastMerge();
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB::ParallelShellSort() {
  FindThreadVariables();

  tbb::task_arena arena(c_threads_);
  arena.execute([&] { tbb::parallel_for(0, c_threads_, [&](int i) { ShellSort(i * mini_batch_); }); });

  Merge();
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB::FindThreadVariables() {
  c_threads_ = tbb::this_task_arena::max_concurrency();
  // Round down to nearest power of 2
  c_threads_ = static_cast<int>(std::pow(2, std::floor(std::log2(c_threads_))));

  n_ = size_ + (((2 * c_threads_) - size_ % (2 * c_threads_))) % (2 * c_threads_);
  mass_.resize(n_, std::numeric_limits<int>::max());
  mini_batch_ = n_ / c_threads_;
  std::ranges::copy(array_ | std::views::take(size_), mass_.begin());
  array_.resize(n_);
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB::RunImpl() {
  ParallelShellSort();
  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_tbb::ShellSortTBB::PostProcessingImpl() {
  int* ptr_ans = reinterpret_cast<int*>(task_data->outputs[0]);

  std::ranges::copy(array_ | std::views::take(size_), ptr_ans);
  return true;
}