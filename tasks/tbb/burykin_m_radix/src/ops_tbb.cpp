#include "tbb/burykin_m_radix/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

std::array<int, 256> burykin_m_radix_tbb::RadixTBB::ComputeFrequencyParallel(const std::vector<int>& a,
                                                                             const int shift) {
  const size_t num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::array<int, 256>> local_counts(num_threads);
  const size_t array_size = a.size();

  tbb::parallel_for(tbb::blocked_range<size_t>(0, array_size), [&](const tbb::blocked_range<size_t>& range) {
    size_t thread_id = tbb::this_task_arena::current_thread_index();
    thread_id = thread_id % num_threads;

    for (size_t i = range.begin(); i < range.end(); ++i) {
      unsigned int key = ((static_cast<unsigned int>(a[i]) >> shift) & 0xFFU);
      if (shift == 24) {
        key ^= 0x80;
      }
      ++local_counts[thread_id][key];
    }
  });

  std::array<int, 256> global_count = {};
  for (size_t t = 0; t < num_threads; ++t) {
    for (int i = 0; i < 256; ++i) {
      global_count[i] += local_counts[t][i];
    }
  }

  return global_count;
}

std::array<int, 256> burykin_m_radix_tbb::RadixTBB::ComputeIndices(const std::array<int, 256>& count) {
  std::array<int, 256> index = {0};
  for (int i = 1; i < 256; ++i) {
    index[i] = index[i - 1] + count[i - 1];
  }
  return index;
}

void burykin_m_radix_tbb::RadixTBB::DistributeElementsParallel(const std::vector<int>& a, std::vector<int>& b,
                                                               const std::array<int, 256>& index, const int shift) {
  const size_t num_threads = ppc::util::GetPPCNumThreads();

  const size_t items_per_thread = (a.size() + num_threads - 1) / num_threads;

  std::vector<std::array<int, 256>> thread_counts(num_threads);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, num_threads), [&](const tbb::blocked_range<size_t>& thread_range) {
    for (size_t t = thread_range.begin(); t < thread_range.end(); ++t) {
      thread_counts[t].fill(0);

      size_t start = t * items_per_thread;
      size_t end = std::min(start + items_per_thread, a.size());

      for (size_t i = start; i < end; ++i) {
        unsigned int key = ((static_cast<unsigned int>(a[i]) >> shift) & 0xFFU);
        if (shift == 24) {
          key ^= 0x80;
        }
        ++thread_counts[t][key];
      }
    }
  });

  std::vector<std::array<int, 256>> thread_offsets(num_threads);
  for (size_t t = 0; t < num_threads; ++t) {
    for (int j = 0; j < 256; ++j) {
      thread_offsets[t][j] = index[j];
      for (size_t prev_t = 0; prev_t < t; ++prev_t) {
        thread_offsets[t][j] += thread_counts[prev_t][j];
      }
    }
  }

  tbb::parallel_for(tbb::blocked_range<size_t>(0, num_threads), [&](const tbb::blocked_range<size_t>& thread_range) {
    for (size_t t = thread_range.begin(); t < thread_range.end(); ++t) {
      size_t start = t * items_per_thread;
      size_t end = std::min(start + items_per_thread, a.size());

      std::array<int, 256> local_offsets = thread_offsets[t];

      for (size_t i = start; i < end; ++i) {
        unsigned int key = ((static_cast<unsigned int>(a[i]) >> shift) & 0xFFU);
        if (shift == 24) {
          key ^= 0x80;
        }
        b[local_offsets[key]++] = a[i];
      }
    }
  });
}

bool burykin_m_radix_tbb::RadixTBB::PreProcessingImpl() {
  const unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  output_.resize(input_size);
  return true;
}

bool burykin_m_radix_tbb::RadixTBB::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool burykin_m_radix_tbb::RadixTBB::RunImpl() {
  if (input_.empty()) {
    return true;
  }

  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());

  arena.execute([&] {
    std::vector<int> a = std::move(input_);
    std::vector<int> b(a.size());

    for (int shift = 0; shift < 32; shift += 8) {
      auto count = ComputeFrequencyParallel(a, shift);
      const auto index = ComputeIndices(count);
      DistributeElementsParallel(a, b, index, shift);
      a.swap(b);
    }

    output_ = std::move(a);
  });

  return true;
}

bool burykin_m_radix_tbb::RadixTBB::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
