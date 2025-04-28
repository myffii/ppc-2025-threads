#include "tbb/kudryashova_i_radix_batcher/include/kudryashovaRadixBatcherTBB.hpp"

#include <oneapi/tbb/combinable.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/partitioner.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

void kudryashova_i_radix_batcher_tbb::ConvertDoublesToUint64(const std::vector<double>& data,
                                                             std::vector<uint64_t>& converted, size_t first) {
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, converted.size()),
      [&](const auto& range) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          uint64_t bits = 0;
          memcpy(&bits, &data[first + i], sizeof(double));
          converted[i] = ((bits & (1ULL << 63)) != 0) ? ~bits : bits ^ (1ULL << 63);
        }
      },
      tbb::auto_partitioner());
}

void kudryashova_i_radix_batcher_tbb::ConvertUint64ToDoubles(std::vector<double>& data,
                                                             const std::vector<uint64_t>& converted, size_t first) {
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, converted.size()),
      [&](const auto& range) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          uint64_t bits = converted[i];
          bits = ((bits & (1ULL << 63)) != 0) ? (bits ^ (1ULL << 63)) : ~bits;
          memcpy(&data[first + i], &bits, sizeof(double));
        }
      },
      tbb::auto_partitioner());
}

void kudryashova_i_radix_batcher_tbb::RadixDoubleSort(std::vector<double>& data, size_t first, size_t last) {
  const size_t sort_size = last - first;
  std::vector<uint64_t> converted(sort_size);
  // Convert each double to uint64_t representation
  ConvertDoublesToUint64(data, converted, first);

  std::vector<uint64_t> buffer(sort_size);
  int bits_int_byte = 8;
  int max_byte_value = 255;
  size_t total_bits = sizeof(uint64_t) * CHAR_BIT;
  for (size_t shift = 0; shift < total_bits; shift += bits_int_byte) {
    tbb::combinable<std::array<size_t, 256>> local_counts;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, sort_size), [&](const auto& range) {
      auto& counts = local_counts.local();
      for (size_t i = range.begin(); i != range.end(); ++i) {
        ++counts[(converted[i] >> shift) & max_byte_value];
      }
    });

    std::array<size_t, 256> total_counts{};
    local_counts.combine_each([&](const auto& local_count) {
      for (size_t i = 0; i < 256; ++i) {
        total_counts[i] += local_count[i];
      }
    });
    // Convert the count array to a prefix sum array
    size_t total = 0;
    for (auto& safe : total_counts) {
      size_t old = safe;
      safe = total;
      total += old;
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, 256), [&](const auto& range) {
      for (size_t j = range.begin(); j != range.end(); ++j) {
        size_t count = total_counts[j];
        for (size_t i = 0; i < sort_size; ++i) {
          if (((converted[i] >> shift) & max_byte_value) == j) {
            buffer[count++] = converted[i];
          }
        }
      }
    });

    converted.swap(buffer);
  }
  ConvertUint64ToDoubles(data, converted, first);
}

void kudryashova_i_radix_batcher_tbb::BatcherMerge(std::vector<double>& target_array, size_t merge_start,
                                                   size_t mid_point, size_t merge_end) {
  const size_t n = merge_end - merge_start;
  if (n <= 1) {
    return;
  }
  tbb::parallel_invoke([&] { BatcherMerge(target_array, merge_start, (merge_start + mid_point) / 2, mid_point); },
                       [&] { BatcherMerge(target_array, mid_point, (mid_point + merge_end) / 2, merge_end); });

  for (size_t step = 1; step < n; step *= 2) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n / (2 * step)), [&](const auto& batch_range) {
      for (size_t i = batch_range.begin(); i != batch_range.end(); ++i) {
        const size_t left = merge_start + (2 * step * i);
        const size_t block_end = std::min(left + (2 * step), merge_end);
        for (size_t j = left + 1; j < block_end; j += 2) {
          if (target_array[j - 1] > target_array[j]) {
            std::swap(target_array[j - 1], target_array[j]);
          }
        }
      }
    });
  }
}

bool kudryashova_i_radix_batcher_tbb::TestTaskTBB::RunImpl() {
  size_t n = input_data_.size();
  RadixDoubleSort(input_data_, 0, n);
  for (size_t step = 1; step < n; step *= 2) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n / (2 * step)), [&](const auto& merge_range) {
      for (size_t i = merge_range.begin(); i != merge_range.end(); ++i) {
        const size_t start = 2 * step * i;
        const size_t mid = start + step;
        const size_t end = std::min(start + (2 * step), n);
        if (mid < end) {
          BatcherMerge(input_data_, start, mid, end);
        }
      }
    });
  }
  return true;
}

bool kudryashova_i_radix_batcher_tbb::TestTaskTBB::PreProcessingImpl() {
  input_data_.resize(task_data->inputs_count[0]);
  if (task_data->inputs[0] == nullptr || task_data->inputs_count[0] == 0) {
    return false;
  }
  auto* tmp_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], input_data_.begin());
  return true;
}

bool kudryashova_i_radix_batcher_tbb::TestTaskTBB::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == task_data->inputs_count[0];
}

bool kudryashova_i_radix_batcher_tbb::TestTaskTBB::PostProcessingImpl() {
  std::ranges::copy(input_data_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
