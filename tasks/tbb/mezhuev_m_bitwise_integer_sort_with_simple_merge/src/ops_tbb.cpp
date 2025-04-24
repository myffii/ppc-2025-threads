#include "tbb/mezhuev_m_bitwise_integer_sort_with_simple_merge/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

namespace mezhuev_m_bitwise_integer_sort_tbb {

namespace {
void SeparateNumbers(const std::vector<int>& input, std::vector<int>& negative, std::vector<int>& positive) {
  for (int num : input) {
    if (num < 0) {
      negative.push_back(-num);
    } else {
      positive.push_back(num);
    }
  }
}

void RadixSort(std::vector<int>& data, int exp) {
  std::vector<int> count(10, 0);
  std::vector<int> output(data.size());

  for (size_t i = 0; i < data.size(); ++i) {
    int digit = (data[i] / exp) % 10;
    count[digit]++;
  }

  for (int i = 1; i < 10; ++i) {
    count[i] += count[i - 1];
  }

  for (size_t i = data.size() - 1; i < data.size(); --i) {
    int digit = (data[i] / exp) % 10;
    output[--count[digit]] = data[i];
  }

  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = output[i];
  }
}

void ProcessNumbers(std::vector<int>& numbers, int max_value) {
  int exp = 1;
  while (max_value / exp > 0) {
    RadixSort(numbers, exp);
    exp *= 10;
  }
}

}  // namespace

bool SortTBB::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);

  input_ = (input_size == 0) ? std::vector<int>() : std::vector<int>(in_ptr, in_ptr + input_size);
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  if (input_size == 0) {
    return true;
  }

  max_value_ = *std::ranges::max_element(input_, [](int a, int b) { return std::abs(a) < std::abs(b); });
  max_value_ = std::abs(max_value_);

  return true;
}

bool SortTBB::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool SortTBB::RunImpl() {
  if (input_.empty()) {
    output_ = input_;
    return true;
  }

  std::vector<int> negative;
  std::vector<int> positive;

  SeparateNumbers(input_, negative, positive);

  oneapi::tbb::task_arena arena(1);
  arena.execute([&] {
    tbb::task_group tg;

    tg.run([&] { ProcessNumbers(positive, max_value_); });
    tg.run([&] { ProcessNumbers(negative, max_value_); });

    tg.wait();
  });

  std::ranges::reverse(negative);
  for (int& num : negative) {
    num = -num;
  }

  output_.clear();
  output_.insert(output_.end(), negative.begin(), negative.end());
  output_.insert(output_.end(), positive.begin(), positive.end());

  return true;
}

bool SortTBB::PostProcessingImpl() {
  if (input_.empty()) {
    return true;
  }

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(output_, out_ptr);
  return true;
}

}  // namespace mezhuev_m_bitwise_integer_sort_tbb
