#include "omp/belov_a_radix_sort_with_batcher_mergesort/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/util/include/util.hpp"

using namespace std;

namespace belov_a_radix_batcher_mergesort_omp {

constexpr int kDecimalBase = 10;

int RadixBatcherMergesortParallel::GetNumberDigitCapacity(Bigint num) {
  return (num == 0 ? 1 : static_cast<int>(log10(abs(num))) + 1);
}

void RadixBatcherMergesortParallel::Sort(std::span<Bigint> arr) {
  vector<Bigint> pos;
  vector<Bigint> neg;

  for (const auto& num : arr) {
    (num >= 0 ? pos : neg).push_back(abs(num));
  }

  RadixSort(pos, false);
  RadixSort(neg, true);

  size_t index = 0;

  for (const auto& num : neg) {
    arr[index++] = -num;
  }

  for (const auto& num : pos) {
    arr[index++] = num;
  }
}

void RadixBatcherMergesortParallel::RadixSort(vector<Bigint>& arr, bool invert) {
  if (arr.empty()) {
    return;
  }

  Bigint max_val = *std::ranges::max_element(arr);
  int max_val_digit_capacity = GetNumberDigitCapacity(max_val);
  int iter = 1;

  for (Bigint digit_place = 1; iter <= max_val_digit_capacity; digit_place *= 10, ++iter) {
    CountingSort(arr, digit_place);
  }

  if (invert) {
    std::ranges::reverse(arr);
  }
}

void RadixBatcherMergesortParallel::CountingSort(vector<Bigint>& arr, Bigint digit_place) {
  vector<Bigint> output(arr.size());
  int count[kDecimalBase] = {};

  for (const auto& num : arr) {
    Bigint index = (num / digit_place) % kDecimalBase;
    count[index]++;
  }

  for (int i = 1; i < kDecimalBase; i++) {
    count[i] += count[i - 1];
  }

  for (size_t i = arr.size() - 1; i < arr.size(); i--) {
    Bigint num = arr[i];
    Bigint index = (num / digit_place) % kDecimalBase;
    output[--count[index]] = num;
  }

  std::ranges::copy(output, arr.begin());
}

void RadixBatcherMergesortParallel::SortParallel(vector<Bigint>& arr) {
  if (arr.empty()) {
    return;
  }

  int num_threads = ppc::util::GetPPCNumThreads();
  size_t chunk_size = arr.size() / num_threads;

#pragma omp parallel num_threads(num_threads)
  {
    int thread_id = omp_get_thread_num();
    size_t start = thread_id * chunk_size;
    size_t end = (thread_id == num_threads - 1) ? arr.size() : start + chunk_size;

    std::span<Bigint> local_span(arr.data() + start, end - start);
    Sort(local_span);
  }
}

void RadixBatcherMergesortParallel::BatcherMergeParallel(vector<Bigint>& arr, int num_threads) {
  size_t n = arr.size();

  if (n == 0) {
    return;
  }

  num_threads = (num_threads < 1) ? 1 : num_threads;

  size_t chunk_size = n / num_threads;              // if n < num_threads, chunk_size = 0
  size_t step = (chunk_size < 1) ? 1 : chunk_size;  // guarantee that step >= 1 (to avoid division by zero)

  for (; step < n; step *= 2) {
#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(n - step); i += static_cast<int64_t>(2 * step)) {
      size_t left = i;
      size_t right = left + step;
      size_t end = (left + (2 * step) < n) ? (left + (2 * step)) : n;

      std::inplace_merge(arr.begin() + static_cast<int64_t>(left), arr.begin() + static_cast<int64_t>(right),
                         arr.begin() + static_cast<int64_t>(end));
    }
  }
}

bool RadixBatcherMergesortParallel::PreProcessingImpl() {
  n_ = task_data->inputs_count[0];
  auto* input_array_data = reinterpret_cast<Bigint*>(task_data->inputs[0]);
  array_.assign(input_array_data, input_array_data + n_);

  return true;
}

bool RadixBatcherMergesortParallel::ValidationImpl() {
  return (task_data->inputs.size() == 1 && !(task_data->inputs_count.size() < 2) && task_data->inputs_count[0] != 0 &&
          (task_data->inputs_count[0] == task_data->inputs_count[1]) && !task_data->outputs.empty());
}

bool RadixBatcherMergesortParallel::RunImpl() {
  int num_threads = ppc::util::GetPPCNumThreads();
  SortParallel(array_);
  BatcherMergeParallel(array_, num_threads);

  return true;
}

bool RadixBatcherMergesortParallel::PostProcessingImpl() {
  std::ranges::copy(array_, reinterpret_cast<Bigint*>(task_data->outputs[0]));
  return true;
}

}  // namespace belov_a_radix_batcher_mergesort_omp