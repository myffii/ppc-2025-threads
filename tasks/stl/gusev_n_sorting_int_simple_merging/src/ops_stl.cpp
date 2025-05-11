#include "stl/gusev_n_sorting_int_simple_merging/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

namespace gusev_n_sorting_int_simple_merging_stl {

void TestTaskSTL::RadixSort(std::vector<int>& arr) {
  if (arr.empty()) {
    return;
  }

  std::vector<int> negatives;
  std::vector<int> positives;
  SplitAndSort(arr, negatives, positives);

  arr.clear();
  arr.insert(arr.end(), negatives.begin(), negatives.end());
  arr.insert(arr.end(), positives.begin(), positives.end());
}

void TestTaskSTL::SplitAndSort(std::vector<int>& arr, std::vector<int>& negatives, std::vector<int>& positives) {
  const size_t data_size = arr.size();
  const int num_threads = (data_size < 1000) ? 1 : ppc::util::GetPPCNumThreads();

  std::vector<std::thread> threads(num_threads);

  std::vector<std::vector<int>> local_negatives(num_threads);
  std::vector<std::vector<int>> local_positives(num_threads);

  size_t total_size = arr.size();
  size_t chunk_size = (total_size + num_threads - 1) / num_threads;

  for (int i = 0; i < num_threads; ++i) {
    size_t start = i * chunk_size;
    size_t end = std::min((i + 1) * chunk_size, total_size);
    if (start >= total_size) {
      break;
    }

    threads[i] = std::thread([&arr, start, end, &local_negatives, &local_positives, i]() {
      for (size_t j = start; j < end; ++j) {
        if (arr[j] < 0) {
          local_negatives[i].push_back(-arr[j]);
        } else {
          local_positives[i].push_back(arr[j]);
        }
      }
    });
  }

  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  for (int i = 0; i < num_threads; ++i) {
    negatives.insert(negatives.end(), local_negatives[i].begin(), local_negatives[i].end());
    positives.insert(positives.end(), local_positives[i].begin(), local_positives[i].end());
  }

  std::thread t_neg([&negatives]() {
    if (!negatives.empty()) {
      RadixSortForNonNegative(negatives);
      std::ranges::reverse(negatives);
      for (auto& num : negatives) {
        num = -num;
      }
    }
  });

  std::thread t_pos([&positives]() {
    if (!positives.empty()) {
      RadixSortForNonNegative(positives);
    }
  });

  t_neg.join();
  t_pos.join();
}

void TestTaskSTL::RadixSortForNonNegative(std::vector<int>& arr) {
  if (arr.empty()) {
    return;
  }

  int max = *std::ranges::max_element(arr);
  for (int exp = 1; max / exp > 0; exp *= 10) {
    CountingSort(arr, exp);
  }
}

void TestTaskSTL::CountingSort(std::vector<int>& arr, int exp) {
  const size_t data_size = arr.size();
  const int num_threads = (data_size < 1000) ? 1 : ppc::util::GetPPCNumThreads();

  std::vector<std::thread> threads;
  std::vector<std::vector<int>> local_counts(num_threads, std::vector<int>(kDigitCount, 0));

  size_t total_size = arr.size();
  size_t chunk_size = (total_size + num_threads - 1) / num_threads;

  for (int i = 0; i < num_threads; ++i) {
    size_t start = i * chunk_size;
    size_t end = std::min((i + 1) * chunk_size, total_size);
    if (start >= total_size) {
      break;
    }

    threads.emplace_back([&arr, exp, start, end, &local_counts, i]() {
      for (size_t j = start; j < end; ++j) {
        int digit = (arr[j] / exp) % kDigitCount;
        local_counts[i][digit]++;
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  std::vector<int> global_count(kDigitCount, 0);
  for (int d = 0; d < kDigitCount; ++d) {
    global_count[d] = std::accumulate(local_counts.begin(), local_counts.end(), 0,
                                      [d](int sum, const std::vector<int>& counts) { return sum + counts[d]; });
  }

  std::partial_sum(global_count.begin(), global_count.end(), global_count.begin());

  std::vector<int> output(arr.size());
  for (size_t i = arr.size(); i > 0; --i) {
    const int digit = (arr[i - 1] / exp) % kDigitCount;
    output[--global_count[digit]] = arr[i - 1];
  }

  arr = std::move(output);
}

bool TestTaskSTL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  output_.resize(input_size);
  return true;
}

bool TestTaskSTL::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool TestTaskSTL::RunImpl() {
  RadixSort(input_);
  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  std::ranges::copy(input_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}

}  // namespace gusev_n_sorting_int_simple_merging_stl
