#include "tbb/Konstantinov_I_Sort_Batcher/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/parallel_invoke.h"

namespace konstantinov_i_sort_batcher_tbb {
namespace {
uint64_t DoubleToKey(double d) {
  uint64_t u = 0;
  std::memcpy(&u, &d, sizeof(d));

  if ((u >> 63) != 0) {
    return ~u;
  }
  return u ^ 0x8000000000000000ULL;
}

double KeyToDouble(uint64_t key) {
  if ((key >> 63) != 0) {
    key = key ^ 0x8000000000000000ULL;
  } else {
    key = ~key;
  }
  double d = NAN;
  std::memcpy(&d, &key, sizeof(d));
  return d;
}

void RadixSorted(std::vector<double>& arr) {
  if (arr.empty()) {
    return;
  }
  size_t n = arr.size();
  std::vector<uint64_t> keys(n);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      keys[i] = DoubleToKey(arr[i]);
    }
  });

  const int radix = 256;
  std::vector<uint64_t> output_keys(n);

  for (int pass = 0; pass < 8; pass++) {
    std::vector<std::atomic<size_t>> count(radix);
    for (auto& c : count) {
      c = 0;
    }
    int shift = pass * 8;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        auto byte = static_cast<uint8_t>((keys[i] >> shift) & 0xFF);
        count[byte].fetch_add(1, std::memory_order_relaxed);
      }
    });

    for (int j = 1; j < radix; j++) {
      count[j] += count[j - 1];
    }

    for (int i = static_cast<int>(n) - 1; i >= 0; i--) {
      auto byte = static_cast<uint8_t>((keys[i] >> shift) & 0xFF);
      output_keys[--count[byte]] = keys[i];
    }
    keys.swap(output_keys);
  }

  tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      arr[i] = KeyToDouble(keys[i]);
    }
  });
}

void BatcherOddEvenMerge(std::vector<double>& arr, int low, int high) {
  if (high - low <= 1) {
    return;
  }
  int mid = (low + high) / 2;

  tbb::parallel_invoke([&] { BatcherOddEvenMerge(arr, low, mid); }, [&] { BatcherOddEvenMerge(arr, mid, high); });

  tbb::parallel_for(tbb::blocked_range<int>(low, mid), [&](const tbb::blocked_range<int>& r) {
    for (int i = r.begin(); i < r.end(); ++i) {
      if (arr[i] > arr[i + mid - low]) {
        std::swap(arr[i], arr[i + mid - low]);
      }
    }
  });
}

void RadixSort(std::vector<double>& arr) {
  RadixSorted(arr);
  BatcherOddEvenMerge(arr, 0, static_cast<int>(arr.size()));
}
}  // namespace
}  // namespace konstantinov_i_sort_batcher_tbb

bool konstantinov_i_sort_batcher_tbb::RadixSortBatcherTBB::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  mas_ = std::vector<double>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0);

  return true;
}

bool konstantinov_i_sort_batcher_tbb::RadixSortBatcherTBB::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool konstantinov_i_sort_batcher_tbb::RadixSortBatcherTBB::RunImpl() {
  output_ = mas_;
  konstantinov_i_sort_batcher_tbb::RadixSort(output_);
  return true;
}

bool konstantinov_i_sort_batcher_tbb::RadixSortBatcherTBB::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}