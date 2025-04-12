#include "omp/malyshev_v_radix_sort/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_v_radix_sort_omp {

RadixSortDoubleOMP::RadixSortDoubleOMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool RadixSortDoubleOMP::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool RadixSortDoubleOMP::PreProcessingImpl() {
  auto* input_data = reinterpret_cast<double*>(task_data->inputs[0]);
  input_ = std::vector<double>(input_data, input_data + task_data->inputs_count[0]);
  output_.resize(task_data->outputs_count[0]);
  return true;
}

void RadixSortDoubleOMP::ConvertDouble(double& val, bool reverse) {
  uint64_t bits = 0;
  memcpy(&bits, &val, sizeof(double));

  if (!reverse) {
    if ((bits & (1ULL << 63)) != 0ULL) {
      bits = ~bits;
    } else {
      bits |= (1ULL << 63);
    }
  } else {
    if ((bits & (1ULL << 63)) != 0ULL) {
      bits &= ~(1ULL << 63);
    } else {
      bits = ~bits;
    }
  }

  memcpy(&val, &bits, sizeof(double));
}

bool RadixSortDoubleOMP::RunImpl() {
  if (input_.empty()) {
    return true;
  }

  for (auto& val : input_) {
    ConvertDouble(val);
  }

  constexpr int kBitsPerPass = 8;
  constexpr int kNumBins = 1 << kBitsPerPass;
  constexpr int kTotalBits = sizeof(double) * 8;

  std::vector<double> buffer(input_.size());

  for (int shift = 0; shift < kTotalBits; shift += kBitsPerPass) {
    std::vector<size_t> count(kNumBins, 0);

#pragma omp parallel
    {
      std::vector<size_t> local_count(kNumBins, 0);
#pragma omp for
      for (int i = 0; i < static_cast<int>(input_.size()); ++i) {
        uint64_t bits = 0;
        memcpy(&bits, &input_[i], sizeof(double));
        uint8_t byte = (bits >> shift) & (kNumBins - 1);
        local_count[byte]++;
      }

#pragma omp critical
      {
        for (int j = 0; j < kNumBins; ++j) {
          count[j] += local_count[j];
        }
      }
    }

    for (size_t i = 1; i < kNumBins; ++i) {
      count[i] += count[i - 1];
    }

    for (int i = static_cast<int>(input_.size()) - 1; i >= 0; --i) {
      uint64_t bits = 0;
      memcpy(&bits, &input_[i], sizeof(double));
      uint8_t byte = (bits >> shift) & (kNumBins - 1);
      size_t idx = --count[byte];
      buffer[idx] = input_[i];
    }

    std::swap(input_, buffer);
  }

  for (auto& val : input_) {
    ConvertDouble(val, true);
  }

  std::ranges::copy(input_, output_.begin());
  return true;
}

bool RadixSortDoubleOMP::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(output_, out_ptr);
  return true;
}

}  // namespace malyshev_v_radix_sort_omp