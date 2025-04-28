#include "tbb/khovansky_d_double_radix_batcher/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/combinable.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace khovansky_d_double_radix_batcher_tbb {
namespace {
uint64_t EncodeDoubleToUint64(double value) {
  uint64_t bit_representation = 0;
  std::memcpy(&bit_representation, &value, sizeof(value));

  if ((bit_representation >> 63) != 0) {
    return ~bit_representation;
  }
  return bit_representation ^ (1ULL << 63);
}

double DecodeUint64ToDouble(uint64_t encoded) {
  if ((encoded >> 63) != 0) {
    encoded ^= (1ULL << 63);
  } else {
    encoded = ~encoded;
  }

  double result = 0.0;
  std::memcpy(&result, &encoded, sizeof(result));
  return result;
}

void RadixSort(std::vector<uint64_t>& array) {
  const int bits_in_byte = 8;
  const int total_bits = 64;
  const int bucket_count = 256;

  std::vector<uint64_t> buffer(array.size(), 0);
  std::vector<int> frequency(bucket_count, 0);

  for (int shift = 0; shift < total_bits; shift += bits_in_byte) {
    tbb::combinable<std::vector<int>> local_freq([&]() { return std::vector<int>(bucket_count, 0); });

    tbb::parallel_for(tbb::blocked_range<size_t>(0, array.size()), [&](const tbb::blocked_range<size_t>& range) {
      auto& local = local_freq.local();
      for (size_t i = range.begin(); i < range.end(); ++i) {
        auto bucket = static_cast<uint8_t>((array[i] >> shift) & 0xFF);
        local[bucket]++;
      }
    });

    std::ranges::fill(frequency, 0);
    local_freq.combine_each([&](const std::vector<int>& local) {
      for (int i = 0; i < bucket_count; ++i) {
        frequency[i] += local[i];
      }
    });

    for (int i = 1; i < bucket_count; ++i) {
      frequency[i] += frequency[i - 1];
    }

    for (int i = static_cast<int>(array.size()) - 1; i >= 0; i--) {
      auto bucket = static_cast<uint8_t>((array[i] >> shift) & 0xFF);
      buffer[--frequency[bucket]] = array[i];
    }

    array.swap(buffer);
  }
}

void BatcherOddEvenMerge(std::vector<uint64_t>& array, int left, int right, int max_depth, int depth = 0) {
  if (right - left <= 1) {
    return;
  }

  int mid = left + ((right - left) / 2);

  if (depth < max_depth) {
    tbb::parallel_invoke([&] { BatcherOddEvenMerge(array, left, mid, depth + 1); },
                         [&] { BatcherOddEvenMerge(array, mid, right, depth + 1); });
  } else {
    BatcherOddEvenMerge(array, left, mid, depth + 1);
    BatcherOddEvenMerge(array, mid, right, depth + 1);
  }

  for (int i = left; i + 1 < right; i += 2) {
    if (array[i] > array[i + 1]) {
      std::swap(array[i], array[i + 1]);
    }
  }
}

void RadixBatcherSort(std::vector<double>& data) {
  std::vector<uint64_t> transformed_data(data.size(), 0);

  int max_parallel_depth = int(std::log2(data.size()) + 1);

  tbb::parallel_for(size_t(0), data.size(), [&](size_t i) { transformed_data[i] = EncodeDoubleToUint64(data[i]); });

  RadixSort(transformed_data);
  BatcherOddEvenMerge(transformed_data, 0, static_cast<int>(transformed_data.size()), max_parallel_depth);

  tbb::parallel_for(size_t(0), data.size(), [&](size_t i) { data[i] = DecodeUint64ToDouble(transformed_data[i]); });
}
}  // namespace
}  // namespace khovansky_d_double_radix_batcher_tbb

bool khovansky_d_double_radix_batcher_tbb::RadixTBB::PreProcessingImpl() {
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);

  unsigned int input_size = task_data->inputs_count[0];
  unsigned int output_size = task_data->outputs_count[0];

  input_ = std::vector<double>(in_ptr, in_ptr + input_size);
  output_ = std::vector<double>(output_size, 0);

  return true;
}

bool khovansky_d_double_radix_batcher_tbb::RadixTBB::ValidationImpl() {
  if (!task_data) {
    return false;
  }

  if (task_data->inputs[0] == nullptr && task_data->inputs_count[0] == 0) {
    return false;
  }

  if (task_data->outputs[0] == nullptr) {
    return false;
  }

  if (task_data->inputs_count[0] < 2) {
    return false;
  }

  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool khovansky_d_double_radix_batcher_tbb::RadixTBB::RunImpl() {
  output_ = input_;
  khovansky_d_double_radix_batcher_tbb::RadixBatcherSort(output_);
  return true;
}

bool khovansky_d_double_radix_batcher_tbb::RadixTBB::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }

  return true;
}
