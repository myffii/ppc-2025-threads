#include "omp/burykin_m_radix/include/ops_omp.hpp"

#include <array>
#include <utility>
#include <vector>

std::array<int, 256> burykin_m_radix_seq::RadixOMP::ComputeFrequency(const std::vector<int>& a, const int shift) {
  std::array<int, 256> count = {};

#pragma omp parallel default(none) shared(a, count, shift)
  {
    // Each thread maintains its own local counter
    std::array<int, 256> local_count = {};

#pragma omp for nowait
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
      const int v = a[i];
      unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
      if (shift == 24) {
        key ^= 0x80;
      }
      ++local_count[key];
    }

// Merge local counters into the shared counter
#pragma omp critical
    {
      for (int i = 0; i < 256; ++i) {
        count[i] += local_count[i];
      }
    }
  }

  return count;
}

std::array<int, 256> burykin_m_radix_seq::RadixOMP::ComputeIndices(const std::array<int, 256>& count) {
  std::array<int, 256> index = {0};
  // This loop has sequential dependency, cannot be parallelized
  for (int i = 1; i < 256; ++i) {
    index[i] = index[i - 1] + count[i - 1];
  }
  return index;
}

void burykin_m_radix_seq::RadixOMP::DistributeElements(const std::vector<int>& a, std::vector<int>& b,
                                                       std::array<int, 256> index, const int shift) {
  // Create a copy of indices for parallel access
  std::array<int, 256> local_index = index;

  // Calculate offset for each element
  std::vector<int> offsets(a.size());

#pragma omp parallel for default(none) shared(a, offsets, local_index, shift)
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    const int v = a[i];
    unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
    if (shift == 24) {
      key ^= 0x80;
    }

    int pos = 0;
#pragma omp critical
    {
      pos = local_index[key];
      local_index[key]++;
    }

    // Store position for later use
    offsets[i] = pos;
  }

// Distribute elements to output array using calculated offsets
#pragma omp parallel for default(none) shared(a, b, offsets)
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    b[offsets[i]] = a[i];
  }
}

bool burykin_m_radix_seq::RadixOMP::PreProcessingImpl() {
  const unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  output_.resize(input_size);
  return true;
}

bool burykin_m_radix_seq::RadixOMP::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool burykin_m_radix_seq::RadixOMP::RunImpl() {
  if (input_.empty()) {
    return true;
  }

  std::vector<int> a = std::move(input_);
  std::vector<int> b(a.size());

#pragma omp parallel
  {
// Single directive ensures one thread executes the outer loop
// while inner operations can still be parallelized
#pragma omp single
    {
      for (int shift = 0; shift < 32; shift += 8) {
        auto count = ComputeFrequency(a, shift);
        const auto index = ComputeIndices(count);
        DistributeElements(a, b, index, shift);
        a.swap(b);
      }
    }
  }

  output_ = std::move(a);
  return true;
}

bool burykin_m_radix_seq::RadixOMP::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  const auto output_size = static_cast<int>(output_.size());

// Parallelize copying results to output buffer
#pragma omp parallel for
  for (int i = 0; i < output_size; ++i) {
    output_ptr[i] = output_[i];
  }
  return true;
}