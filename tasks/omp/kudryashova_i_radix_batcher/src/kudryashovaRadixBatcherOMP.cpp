#include "omp/kudryashova_i_radix_batcher/include/kudryashovaRadixBatcherOMP.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

void kudryashova_i_radix_batcher_omp::RadixDoubleSort(std::vector<double>& data, size_t first, size_t last) {
  const size_t sort_size = last - first;
  std::vector<uint64_t> converted(sort_size);

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(sort_size); ++i) {
    double value = data[first + i];
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(value));
    converted[i] = ((bits & (1ULL << 63)) != 0) ? ~bits : bits ^ (1ULL << 63);
  }

  std::vector<uint64_t> buffer(sort_size);
  const int bits_in_byte = 8;
  const int total_passes = sizeof(uint64_t);
  const int max_byte_value = 255;

  for (int shift = 0; shift < total_passes; ++shift) {
    std::vector<size_t> count(256, 0);
    const int shift_loc = shift * bits_in_byte;

#pragma omp parallel
    {
      std::vector<size_t> local_count(256, 0);
#pragma omp for
      for (int i = 0; i < static_cast<int>(sort_size); ++i) {
        ++local_count[(converted[i] >> shift_loc) & max_byte_value];
      }
#pragma omp critical
      {
        for (int j = 0; j < 256; ++j) {
          count[j] += local_count[j];
        }
      }
    }

    size_t total = 0;
    for (int j = 0; j < 256; ++j) {
      size_t old = count[j];
      count[j] = total;
      total += old;
    }

    std::vector<std::atomic<size_t>> atomic_count(256);
    for (int j = 0; j < 256; ++j) {
      atomic_count[j] = count[j];
    }
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(sort_size); ++i) {
      const uint8_t byte = (converted[i] >> shift_loc) & max_byte_value;
      size_t pos = atomic_count[byte].fetch_add(1, std::memory_order_seq_cst);
      buffer[pos] = converted[i];
    }

    converted.swap(buffer);
  }

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(sort_size); ++i) {
    uint64_t bits = converted[i];
    bits = ((bits & (1ULL << 63)) != 0) ? (bits ^ (1ULL << 63)) : ~bits;
    std::memcpy(&data[first + i], &bits, sizeof(double));
  }
}

void kudryashova_i_radix_batcher_omp::BatcherMerge(std::vector<double>& target_array, size_t merge_start,
                                                   size_t mid_point, size_t merge_end) {
  const size_t total_elements = merge_end - merge_start;
  std::vector<double> merge_buffer(total_elements);
  const size_t left_size = mid_point - merge_start;
  const size_t right_size = merge_end - mid_point;
  std::vector<double> left_array(target_array.begin() + static_cast<std::ptrdiff_t>(merge_start),
                                 target_array.begin() + static_cast<std::ptrdiff_t>(mid_point));

  std::vector<double> right_array(target_array.begin() + static_cast<std::ptrdiff_t>(mid_point),
                                  target_array.begin() + static_cast<std::ptrdiff_t>(merge_end));
  size_t left_ptr = 0;
  size_t right_ptr = 0;
  size_t merge_ptr = merge_start;
#pragma omp parallel
  {
    const int num_threads = omp_get_num_threads();
    const int thread_id = omp_get_thread_num();
    const size_t chunk = (total_elements + num_threads - 1) / num_threads;
    const size_t start = std::min(thread_id * chunk, total_elements);
    const size_t end = std::min((thread_id + 1) * chunk, total_elements);

    for (size_t i = start; i < end; ++i) {
      if (i % 2 == 0) {
        if (left_ptr < left_size && (right_ptr >= right_size || left_array[left_ptr] <= right_array[right_ptr])) {
          target_array[merge_ptr++] = left_array[left_ptr++];
        } else {
          target_array[merge_ptr++] = right_array[right_ptr++];
        }
      } else {
        if (right_ptr < right_size && (left_ptr >= left_size || right_array[right_ptr] <= left_array[left_ptr])) {
          target_array[merge_ptr++] = right_array[right_ptr++];
        } else {
          target_array[merge_ptr++] = left_array[left_ptr++];
        }
      }
    }
  }
}

bool kudryashova_i_radix_batcher_omp::TestTaskOpenMP::PreProcessingImpl() {
  input_data_.resize(task_data->inputs_count[0]);
  if (task_data->inputs[0] == nullptr || task_data->inputs_count[0] == 0) {
    return false;
  }
  auto* tmp_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], input_data_.begin());
  return true;
}

bool kudryashova_i_radix_batcher_omp::TestTaskOpenMP::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == task_data->inputs_count[0];
}

bool kudryashova_i_radix_batcher_omp::TestTaskOpenMP::RunImpl() {
  size_t n = input_data_.size();
  const int num_threads = omp_get_max_threads();
  const size_t block_size = (n + num_threads - 1) / num_threads;

#pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; ++i) {
    const size_t start = i * block_size;
    const size_t end = std::min(start + block_size, n);
    if (start < n) {
      RadixDoubleSort(input_data_, start, end);
    }
  }

  for (size_t merge_size = block_size; merge_size < n; merge_size *= 2) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(n); i += static_cast<int>(2 * merge_size)) {
      const size_t mid = std::min(i + merge_size, n);
      const size_t end = std::min(i + (2 * merge_size), n);
      if (mid < end) {
        BatcherMerge(input_data_, i, mid, end);
      }
    }
  }
  return true;
}

bool kudryashova_i_radix_batcher_omp::TestTaskOpenMP::PostProcessingImpl() {
  std::ranges::copy(input_data_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
