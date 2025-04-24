#include "tbb/tsatsyn_a_radix_sort_simple_merge/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/combinable.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
inline int CalculateBits(const std::vector<uint64_t>& data, bool is_pozitive) {
  if (data.empty()) {
    return 0;
  }
  uint64_t extreme_val = 0;
  int num_bits = 0;
  if (is_pozitive) {
    extreme_val = *std::ranges::max_element(data);
    num_bits = std::bit_width(extreme_val);
  } else {
    extreme_val = *std::ranges::min_element(data);
    num_bits = (extreme_val == 0) ? 0 : std::bit_width(extreme_val);
  }

  return num_bits;
}
}  // namespace
bool tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB::PreProcessingImpl() {
  // Init value for input and output
  auto* temp_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_data_ = std::vector<double>(temp_ptr, temp_ptr + task_data->inputs_count[0]);
  output_.resize(task_data->inputs_count[0]);
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB::ValidationImpl() {
  // Check equality of counts elements
  return (task_data->inputs_count[0] != 0) && (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB::RunImpl() {
  std::vector<uint64_t> pozitive_copy;
  std::vector<uint64_t> negative_copy;
  tbb::combinable<std::vector<uint64_t>> pos_comb;
  tbb::combinable<std::vector<uint64_t>> neg_comb;
  tbb::parallel_for(tbb::blocked_range<size_t>(0, input_data_.size()), [&](const auto& r) {
    auto& pos = pos_comb.local();
    auto& neg = neg_comb.local();
    for (size_t i = r.begin(); i < r.end(); ++i) {
      double val = input_data_[i];
      auto* bits = reinterpret_cast<uint64_t*>(&val);
      (val > 0.0 ? pos : neg).push_back(*bits);
    }
  });
  auto merge_vectors = [](auto& comb) {
    std::vector<uint64_t> res;
    comb.combine_each([&](const auto& v) { res.insert(res.end(), v.begin(), v.end()); });
    return res;
  };

  pozitive_copy = merge_vectors(pos_comb);
  negative_copy = merge_vectors(neg_comb);
  int pozitive_bits = CalculateBits(pozitive_copy, true);
  int negative_bits = CalculateBits(negative_copy, false);
  for (int bit = 0; bit < pozitive_bits; bit++) {
    std::vector<uint64_t> group0;
    std::vector<uint64_t> group1;
    for (uint64_t b : pozitive_copy) {
      if (((b >> bit) & 1) != 0U) {
        group1.push_back(b);
      } else {
        group0.push_back(b);
      }
    }
    pozitive_copy.clear();
    pozitive_copy.insert(pozitive_copy.end(), group0.begin(), group0.end());
    pozitive_copy.insert(pozitive_copy.end(), group1.begin(), group1.end());
  }

  for (int bit = 0; bit < negative_bits; bit++) {
    std::vector<uint64_t> group0;
    std::vector<uint64_t> group1;
    for (uint64_t b : negative_copy) {
      if (((b >> bit) & 1) != 0U) {
        group1.push_back(b);
      } else {
        group0.push_back(b);
      }
    }
    negative_copy.clear();
    negative_copy.insert(negative_copy.end(), group1.begin(), group1.end());
    negative_copy.insert(negative_copy.end(), group0.begin(), group0.end());
  }
  tbb::parallel_invoke(
      [&] {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, negative_copy.size()), [&](const auto& r) {
          for (size_t i = r.begin(); i < r.end(); ++i) {
            double tmp = NAN;
            memcpy(&tmp, &negative_copy[i], sizeof(tmp));
            output_[i] = tmp;
          }
        });
      },
      [&] {
        const size_t offset = negative_copy.size();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, pozitive_copy.size()), [&](const auto& r) {
          for (size_t i = r.begin(); i < r.end(); ++i) {
            double tmp = NAN;
            memcpy(&tmp, &pozitive_copy[i], sizeof(tmp));
            output_[offset + i] = tmp;
          }
        });
      });
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
