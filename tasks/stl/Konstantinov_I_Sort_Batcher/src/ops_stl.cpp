#include "stl/Konstantinov_I_Sort_Batcher/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace konstantinov_i_sort_batcher_stl {
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
void ParallelConvertToKeys(std::vector<double>& arr, std::vector<uint64_t>& keys, int thread_count) {
  size_t n = arr.size();
  size_t block_size = (n + thread_count - 1) / thread_count;
  std::vector<std::thread> threads(thread_count);

  for (int t = 0; t < thread_count; ++t) {
    threads[t] = std::thread([&arr, &keys, t, block_size, n]() {
      size_t begin = t * block_size;
      size_t end = std::min(begin + block_size, n);
      for (size_t i = begin; i < end; ++i) {
        keys[i] = DoubleToKey(arr[i]);
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }
}
void RadixPass(std::vector<uint64_t>& keys, std::vector<uint64_t>& output_keys, int pass, int thread_count) {
  size_t n = keys.size();
  size_t block_size = (n + thread_count - 1) / thread_count;
  int shift = pass * 8;
  const int radix = 256;

  std::vector<std::vector<size_t>> local_counts(thread_count, std::vector<size_t>(radix, 0));
  std::vector<std::thread> threads(thread_count);

  for (int t = 0; t < thread_count; ++t) {
    threads[t] = std::thread([t, &keys, &local_counts, shift, block_size, n]() {
      size_t begin = t * block_size;
      size_t end = std::min(begin + block_size, n);
      for (size_t i = begin; i < end; ++i) {
        auto byte = static_cast<uint8_t>((keys[i] >> shift) & 0xFF);
        local_counts[t][byte]++;
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }
  std::vector<size_t> count(radix, 0);
  for (int b = 0; b < radix; ++b) {
    for (int t = 0; t < thread_count; ++t) {
      count[b] += local_counts[t][b];
    }
  }
  for (int i = 1; i < radix; ++i) {
    count[i] += count[i - 1];
  }
  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    auto byte = static_cast<uint8_t>((keys[i] >> shift) & 0xFF);
    output_keys[--count[byte]] = keys[i];
  }
}
void ParallelConvertBack(std::vector<uint64_t>& keys, std::vector<double>& arr, int thread_count) {
  size_t n = keys.size();
  size_t block_size = (n + thread_count - 1) / thread_count;
  std::vector<std::thread> threads(thread_count);

  for (int t = 0; t < thread_count; ++t) {
    threads[t] = std::thread([&arr, &keys, t, block_size, n]() {
      size_t begin = t * block_size;
      size_t end = std::min(begin + block_size, n);
      for (size_t i = begin; i < end; ++i) {
        arr[i] = KeyToDouble(keys[i]);
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }
}

void RadixSorted(std::vector<double>& arr) {
  if (arr.empty()) {
    return;
  }
  size_t n = arr.size();
  std::vector<uint64_t> keys(n);
  const int thread_count = ppc::util::GetPPCNumThreads();

  ParallelConvertToKeys(arr, keys, thread_count);

  std::vector<uint64_t> output_keys(n);
  for (int pass = 0; pass < 8; ++pass) {
    RadixPass(keys, output_keys, pass, thread_count);
    keys.swap(output_keys);
  }

  ParallelConvertBack(keys, arr, thread_count);
}

void BatcherOddEvenMerge(std::vector<double>& arr, int low, int high) {
  if (high - low <= 1) {
    return;
  }
  int mid = (low + high) / 2;

  BatcherOddEvenMerge(arr, low, mid);
  BatcherOddEvenMerge(arr, mid, high);

  for (int i = low; i < mid; ++i) {
    if (arr[i] > arr[i + mid - low]) {
      std::swap(arr[i], arr[i + mid - low]);
    }
  }
}

void RadixSort(std::vector<double>& arr) {
  RadixSorted(arr);
  BatcherOddEvenMerge(arr, 0, static_cast<int>(arr.size()));
}
}  // namespace
}  // namespace konstantinov_i_sort_batcher_stl

bool konstantinov_i_sort_batcher_stl::RadixSortBatcherSTL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  mas_ = std::vector<double>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0);

  return true;
}

bool konstantinov_i_sort_batcher_stl::RadixSortBatcherSTL::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool konstantinov_i_sort_batcher_stl::RadixSortBatcherSTL::RunImpl() {
  output_ = mas_;
  konstantinov_i_sort_batcher_stl::RadixSort(output_);
  return true;
}

bool konstantinov_i_sort_batcher_stl::RadixSortBatcherSTL::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}