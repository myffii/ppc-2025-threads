#include "stl/kudryashova_i_radix_batcher/include/kudryashovaRadixBatcherSTL.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

void kudryashova_i_radix_batcher_stl::RadixDoubleSort(std::vector<double> &data, size_t first, size_t last) {
  const size_t sort_size = last - first;
  if (sort_size == 0) {
    return;
  }
  std::vector<uint64_t> converted;
  converted.reserve(sort_size);
  for (size_t i = 0; i < sort_size; ++i) {
    double value = data[first + i];
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(value));
    converted.push_back(((bits & (1ULL << 63)) != 0) ? ~bits : bits ^ (1ULL << 63));
  }
  std::vector<uint64_t> buffer;
  buffer.reserve(sort_size);
  buffer.resize(sort_size);
  int bits_int_byte = 8;
  int total_passes = sizeof(uint64_t);
  int max_byte_value = 255;

  for (int shift = 0; shift < total_passes; ++shift) {
    std::array<size_t, 256> count = {0};
    const int shift_loc = shift * bits_int_byte;
    for (const auto &num : converted) {
      ++count[(num >> shift_loc) & max_byte_value];
    }
    size_t total = 0;
    for (auto &safe : count) {
      size_t old = safe;
      safe = total;
      total += old;
    }
    for (const auto &num : converted) {
      const uint8_t byte = (num >> shift_loc) & max_byte_value;
      buffer[count[byte]++] = num;
    }
    converted.swap(buffer);
  }
  for (size_t i = 0; i < sort_size; ++i) {
    uint64_t bits = converted[i];
    bits = ((bits & (1ULL << 63)) != 0) ? (bits ^ (1ULL << 63)) : ~bits;
    std::memcpy(&data[first + i], &bits, sizeof(double));
  }
}
void kudryashova_i_radix_batcher_stl::BatcherMerge(std::vector<double> &target_array, size_t merge_start,
                                                   size_t mid_point, size_t merge_end) {
  const size_t total_elements = merge_end - merge_start;
  const size_t left_size = mid_point - merge_start;
  const size_t right_size = merge_end - mid_point;
  std::vector<double> left_array(target_array.begin() + static_cast<std::ptrdiff_t>(merge_start),
                                 target_array.begin() + static_cast<std::ptrdiff_t>(mid_point));
  std::vector<double> right_array(target_array.begin() + static_cast<std::ptrdiff_t>(mid_point),
                                  target_array.begin() + static_cast<std::ptrdiff_t>(merge_end));
  size_t left_ptr = 0;
  size_t right_ptr = 0;
  size_t merge_ptr = merge_start;

  for (size_t i = 0; i < total_elements; ++i) {
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

bool kudryashova_i_radix_batcher_stl::TestTaskSTL::RunImpl() {
  const size_t input_size = input_data_.size();
  if (input_size <= 1) {
    return true;
  }

  const size_t num_threads = ppc::util::GetPPCNumThreads();
  const size_t sort_block_size = (input_size + num_threads - 1) / num_threads;

  std::vector<std::thread> sort_threads;
  for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    const size_t block_start = thread_idx * sort_block_size;
    const size_t block_end = std::min(block_start + sort_block_size, input_size);

    if (block_start >= input_size) {
      break;
    }

    sort_threads.emplace_back([this, block_start, block_end] { RadixDoubleSort(input_data_, block_start, block_end); });
  }
  for (auto &thread : sort_threads) {
    thread.join();
  }

  for (size_t current_merge_size = sort_block_size; current_merge_size < input_size; current_merge_size *= 2) {
    const size_t merge_group_size = 2 * current_merge_size;
    const size_t total_merge_groups = (input_size + merge_group_size - 1) / merge_group_size;

    std::vector<std::thread> merge_threads;
    const size_t merge_groups_per_thread = (total_merge_groups + num_threads - 1) / num_threads;

    for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
      const size_t group_range_start = thread_idx * merge_groups_per_thread;
      const size_t group_range_end = std::min(group_range_start + merge_groups_per_thread, total_merge_groups);

      if (group_range_start >= group_range_end) {
        break;
      }
      merge_threads.emplace_back([&, group_range_start, group_range_end, current_merge_size, merge_group_size] {
        for (size_t group_index = group_range_start; group_index < group_range_end; ++group_index) {
          const size_t merge_start = group_index * merge_group_size;
          const size_t merge_mid = std::min(merge_start + current_merge_size, input_size);
          const size_t merge_end = std::min(merge_start + merge_group_size, input_size);
          if (merge_mid < merge_end) {
            BatcherMerge(input_data_, merge_start, merge_mid, merge_end);
          }
        }
      });
    }
    for (auto &thread : merge_threads) {
      thread.join();
    }
  }
  return true;
}

bool kudryashova_i_radix_batcher_stl::TestTaskSTL::PreProcessingImpl() {
  input_data_.resize(task_data->inputs_count[0]);
  if (task_data->inputs[0] == nullptr || task_data->inputs_count[0] == 0) {
    return false;
  }
  auto *tmp_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], input_data_.begin());
  return true;
}

bool kudryashova_i_radix_batcher_stl::TestTaskSTL::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == task_data->inputs_count[0];
}

bool kudryashova_i_radix_batcher_stl::TestTaskSTL::PostProcessingImpl() {
  std::ranges::copy(input_data_, reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
