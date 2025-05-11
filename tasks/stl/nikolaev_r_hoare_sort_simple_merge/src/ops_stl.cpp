#include "../include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <random>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool nikolaev_r_hoare_sort_simple_merge_stl::HoareSortSimpleMergeSTL::PreProcessingImpl() {
  vect_size_ = task_data->inputs_count[0];
  auto *vect_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  vect_ = std::vector<double>(vect_ptr, vect_ptr + vect_size_);

  return true;
}

bool nikolaev_r_hoare_sort_simple_merge_stl::HoareSortSimpleMergeSTL::ValidationImpl() {
  return task_data->inputs_count[0] != 0 && task_data->outputs_count[0] != 0 && task_data->inputs[0] != nullptr &&
         task_data->outputs[0] != nullptr && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool nikolaev_r_hoare_sort_simple_merge_stl::HoareSortSimpleMergeSTL::RunImpl() {
  unsigned int num_threads = ppc::util::GetPPCNumThreads();
  if (num_threads == 1 || vect_size_ < num_threads) {
    QuickSort(0, vect_size_ - 1);
    return true;
  }

  size_t total = vect_.size();
  std::vector<size_t> boundaries;
  boundaries.push_back(0);
  size_t chunk_size = total / num_threads;
  for (unsigned int i = 1; i < num_threads; i++) {
    boundaries.push_back(i * chunk_size);
  }
  boundaries.push_back(total);

  std::vector<std::thread> threads;
  for (unsigned int i = 0; i < num_threads; ++i) {
    size_t start_index = boundaries[i];
    size_t end_index = boundaries[i + 1];
    threads.emplace_back([this, start_index, end_index]() { QuickSort(start_index, end_index - 1); });
  }

  for (auto &t : threads) {
    t.join();
  }

  size_t merged_end = boundaries[1];
  for (unsigned int i = 1; i < num_threads; i++) {
    size_t next_end = boundaries[i + 1];
    std::inplace_merge(vect_.begin(), vect_.begin() + static_cast<std::vector<double>::difference_type>(merged_end),
                       vect_.begin() + static_cast<std::vector<double>::difference_type>(next_end));
    merged_end = next_end;
  }
  return true;
}

bool nikolaev_r_hoare_sort_simple_merge_stl::HoareSortSimpleMergeSTL::PostProcessingImpl() {
  for (size_t i = 0; i < vect_size_; i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = vect_[i];
  }
  return true;
}

size_t nikolaev_r_hoare_sort_simple_merge_stl::HoareSortSimpleMergeSTL::Partition(size_t low, size_t high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(static_cast<int>(low), static_cast<int>(high));

  size_t random_pivot_index = dist(gen);
  double pivot = vect_[random_pivot_index];

  std::swap(vect_[random_pivot_index], vect_[low]);
  size_t i = low + 1;

  for (size_t j = low + 1; j <= high; ++j) {
    if (vect_[j] < pivot) {
      std::swap(vect_[i], vect_[j]);
      i++;
    }
  }

  std::swap(vect_[low], vect_[i - 1]);
  return i - 1;
}

void nikolaev_r_hoare_sort_simple_merge_stl::HoareSortSimpleMergeSTL::QuickSort(size_t low, size_t high) {
  if (low >= high) {
    return;
  }
  size_t pivot = Partition(low, high);
  if (pivot > low) {
    QuickSort(low, pivot - 1);
  }
  QuickSort(pivot + 1, high);
}
