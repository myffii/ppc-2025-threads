#include "stl/volochaev_s_Shell_sort_with_Batchers_even-odd_merge/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <ranges>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_stl::ShellSortSTL::PreProcessingImpl() {
  // Init value for input and output
  size_ = static_cast<int>(task_data->inputs_count[0]);

  auto* input_pointer = reinterpret_cast<int*>(task_data->inputs[0]);
  array_ = std::vector<int>(input_pointer, input_pointer + size_);
  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_stl::ShellSortSTL::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_stl::ShellSortSTL::ShellSort(int start) {
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

void volochaev_s_shell_sort_with_batchers_even_odd_merge_stl::ShellSortSTL::MergeBlocks(int id_l, int id_r, int len) {
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

void volochaev_s_shell_sort_with_batchers_even_odd_merge_stl::ShellSortSTL::LastMerge() {
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

void volochaev_s_shell_sort_with_batchers_even_odd_merge_stl::ShellSortSTL::Merge() {
  int current_threads = c_threads_;

  while (current_threads > 1) {
    std::vector<std::future<void>> futures;
    int l = mini_batch_ * (c_threads_ / current_threads);

    for (int i = 0; i < current_threads / 2; ++i) {
      futures.emplace_back(
          std::async(std::launch::async, [this, i, l]() { MergeBlocks((i * 2 * l), (i * 2 * l) + l, l); }));

      futures.emplace_back(
          std::async(std::launch::async, [this, i, l]() { MergeBlocks((i * 2 * l) + 1, (i * 2 * l) + l + 1, l - 1); }));
    }

    for (auto& future : futures) {
      future.get();
    }

    current_threads /= 2;
  }

  LastMerge();
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_stl::ShellSortSTL::ParallelShellSort() {
  FindThreadVariables();

  std::vector<std::future<void>> futures;
  futures.reserve(c_threads_);

  for (int i = 0; i < c_threads_; ++i) {
    futures.emplace_back(std::async(std::launch::async, [this, i]() { ShellSort(i * mini_batch_); }));
  }

  for (auto& future : futures) {
    future.get();
  }

  Merge();
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_stl::ShellSortSTL::FindThreadVariables() {
  auto max_threads =
      std::min(static_cast<unsigned int>(ppc::util::GetPPCNumThreads()), std::thread::hardware_concurrency());
  c_threads_ = static_cast<int>(std::pow(2, std::floor(std::log2(max_threads))));
  n_ = size_ + (((2 * c_threads_) - size_ % (2 * c_threads_))) % (2 * c_threads_);
  mass_.resize(n_, std::numeric_limits<int>::max());
  mini_batch_ = n_ / c_threads_;
  std::ranges::copy(array_ | std::views::take(size_), mass_.begin());
  array_.resize(n_);
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_stl::ShellSortSTL::RunImpl() {
  ParallelShellSort();
  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_stl::ShellSortSTL::PostProcessingImpl() {
  int* ptr_ans = reinterpret_cast<int*>(task_data->outputs[0]);

  std::ranges::copy(array_ | std::views::take(size_), ptr_ans);
  return true;
}
