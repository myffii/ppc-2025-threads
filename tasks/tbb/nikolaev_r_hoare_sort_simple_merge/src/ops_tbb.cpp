#include "tbb/nikolaev_r_hoare_sort_simple_merge/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

bool nikolaev_r_hoare_sort_simple_merge_tbb::HoareSortSimpleMergeTBB::PreProcessingImpl() {
  vect_size_ = task_data->inputs_count[0];
  auto *vect_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  vect_ = std::vector<double>(vect_ptr, vect_ptr + vect_size_);

  return true;
}

bool nikolaev_r_hoare_sort_simple_merge_tbb::HoareSortSimpleMergeTBB::ValidationImpl() {
  return task_data->inputs_count[0] != 0 && task_data->outputs_count[0] != 0 && task_data->inputs[0] != nullptr &&
         task_data->outputs[0] != nullptr && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool nikolaev_r_hoare_sort_simple_merge_tbb::HoareSortSimpleMergeTBB::RunImpl() {
  size_t num_segments = ppc::util::GetPPCNumThreads();

  num_segments = std::min(num_segments, vect_size_);

  size_t segment_size = vect_size_ / num_segments;
  size_t remainder = vect_size_ % num_segments;

  std::vector<std::pair<size_t, size_t>> segments;
  size_t start = 0;

  for (size_t i = 0; i < num_segments; ++i) {
    size_t current_segment_size = segment_size + (i < remainder ? 1 : 0);
    size_t end = start + current_segment_size - 1;
    segments.emplace_back(start, end);
    start = end + 1;
  }

  oneapi::tbb::task_arena arena(static_cast<int>(num_segments));
  arena.execute([this, &segments]() {
    oneapi::tbb::task_group tg;

    for (const auto &seg : segments) {
      tg.run([this, seg]() { QuickSort(seg.first, seg.second); });
    }

    tg.wait();
  });

  size_t merged_end = segments[0].second;
  for (size_t i = 1; i < segments.size(); ++i) {
    std::inplace_merge(vect_.begin(), vect_.begin() + static_cast<std::vector<double>::difference_type>(merged_end + 1),
                       vect_.begin() + static_cast<std::vector<double>::difference_type>(segments[i].second + 1));
    merged_end = segments[i].second;
  }

  return true;
}

bool nikolaev_r_hoare_sort_simple_merge_tbb::HoareSortSimpleMergeTBB::PostProcessingImpl() {
  for (size_t i = 0; i < vect_size_; i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = vect_[i];
  }
  return true;
}

size_t nikolaev_r_hoare_sort_simple_merge_tbb::HoareSortSimpleMergeTBB::Partition(size_t low, size_t high) {
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

void nikolaev_r_hoare_sort_simple_merge_tbb::HoareSortSimpleMergeTBB::QuickSort(size_t low, size_t high) {
  if (low >= high) {
    return;
  }
  size_t pivot = Partition(low, high);
  if (pivot > low) {
    QuickSort(low, pivot - 1);
  }
  QuickSort(pivot + 1, high);
}
