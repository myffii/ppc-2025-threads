#include "tbb/opolin_d_radix_sort_batcher_merge/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/enumerable_thread_specific.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/parallel_invoke.h"

bool opolin_d_radix_batcher_sort_tbb::RadixBatcherSortTaskTbb::PreProcessingImpl() {
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + size_);
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);
  return true;
}

bool opolin_d_radix_batcher_sort_tbb::RadixBatcherSortTaskTbb::ValidationImpl() {
  // Check equality of counts elements
  size_ = static_cast<int>(task_data->inputs_count[0]);
  if (size_ <= 0 || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool opolin_d_radix_batcher_sort_tbb::RadixBatcherSortTaskTbb::RunImpl() {
  std::vector<uint32_t> keys(size_);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, size_), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      keys[i] = ConvertIntToUint(input_[i]);
    }
  });
  RadixSort(keys);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, size_), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      output_[i] = ConvertUintToInt(keys[i]);
    }
  });
  BatcherOddEvenMerge(output_, 0, static_cast<int>(size_));
  return true;
}

bool opolin_d_radix_batcher_sort_tbb::RadixBatcherSortTaskTbb::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

uint32_t opolin_d_radix_batcher_sort_tbb::ConvertIntToUint(int num) { return static_cast<uint32_t>(num) ^ 0x80000000U; }

int opolin_d_radix_batcher_sort_tbb::ConvertUintToInt(uint32_t unum) { return static_cast<int>(unum ^ 0x80000000U); }

void opolin_d_radix_batcher_sort_tbb::RadixSort(std::vector<uint32_t>& uns_vec) {
  size_t sz = uns_vec.size();
  if (sz <= 1) {
    return;
  }
  const int rad = 256;
  std::vector<uint32_t> res(sz);
  for (int stage = 0; stage < 4; stage++) {
    tbb::enumerable_thread_specific<std::vector<size_t>> local_counts([] { return std::vector<size_t>(rad, 0); });
    int shift = stage * 8;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, sz), [&](const tbb::blocked_range<size_t>& r) {
      auto& lc = local_counts.local();
      for (size_t i = r.begin(); i < r.end(); ++i) {
        const uint8_t byte = (uns_vec[i] >> shift) & 255;
        lc[byte]++;
      }
    });
    std::vector<size_t> pref(rad, 0);
    for (auto& lc : local_counts) {
      for (int j = 0; j < rad; ++j) {
        pref[j] += lc[j];
      }
    }
    for (int j = 1; j < rad; ++j) {
      pref[j] += pref[j - 1];
    }
    for (int i = static_cast<int>(sz) - 1; i >= 0; --i) {
      const uint8_t byte = (uns_vec[i] >> shift) & 255;
      res[--pref[byte]] = uns_vec[i];
    }
    uns_vec.swap(res);
  }
}

void opolin_d_radix_batcher_sort_tbb::BatcherOddEvenMerge(std::vector<int>& vec, int low, int high) {
  if (high - low <= 1) {
    return;
  }
  int mid = (low + high) / 2;
  tbb::parallel_invoke([&] { BatcherOddEvenMerge(vec, low, mid); }, [&] { BatcherOddEvenMerge(vec, mid, high); });
  tbb::parallel_for(tbb::blocked_range<int>(low, mid), [&](const tbb::blocked_range<int>& r) {
    for (int i = r.begin(); i < r.end(); ++i) {
      if (vec[i] > vec[i + mid - low]) {
        std::swap(vec[i], vec[i + mid - low]);
      }
    }
  });
}