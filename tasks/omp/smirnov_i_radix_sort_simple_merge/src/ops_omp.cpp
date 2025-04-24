#include "omp/smirnov_i_radix_sort_simple_merge/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <numeric>
#include <utility>
#include <vector>

std::vector<int> smirnov_i_radix_sort_simple_merge_omp::TestTaskOpenMP::Merge(std::vector<int> mas1,
                                                                              std::vector<int> mas2) {
  std::vector<int> res;
  res.reserve(mas1.size() + mas2.size());
  int p1 = 0;
  int p2 = 0;
  while (static_cast<int>(mas1.size()) != p1 && static_cast<int>(mas2.size()) != p2) {
    if (mas1[p1] < mas2[p2]) {
      res.push_back(mas1[p1]);
      p1++;
    } else if (mas2[p2] < mas1[p1]) {
      res.push_back(mas2[p2]);
      p2++;
    } else {
      res.push_back(mas1[p1]);
      res.push_back(mas2[p2]);
      p1++;
      p2++;
    }
  }
  while (static_cast<int>(mas1.size()) != p1) {
    res.push_back(mas1[p1]);
    p1++;
  }
  while (static_cast<int>(mas2.size()) != p2) {
    res.push_back(mas2[p2]);
    p2++;
  }
  return res;
}
void smirnov_i_radix_sort_simple_merge_omp::TestTaskOpenMP::RadixSort(std::vector<int>& mas) {
  if (mas.empty()) {
    return;
  }
  int longest = *std::ranges::max_element(mas.begin(), mas.end());
  int len = std::ceil(std::log10(longest + 1));
  std::vector<int> sorting(mas.size());
  int base = 1;
  for (int j = 0; j < len; j++, base *= 10) {
    std::vector<int> counting(10, 0);
    for (size_t i = 0; i < mas.size(); i++) {
      counting[mas[i] / base % 10]++;
    }
    std::partial_sum(counting.begin(), counting.end(), counting.begin());
    for (int i = static_cast<int>(mas.size() - 1); i >= 0; i--) {
      int pos = counting[mas[i] / base % 10] - 1;
      sorting[pos] = mas[i];
      counting[mas[i] / base % 10]--;
    }
    std::swap(mas, sorting);
  }
}
bool smirnov_i_radix_sort_simple_merge_omp::TestTaskOpenMP::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  mas_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);
  return true;
}
bool smirnov_i_radix_sort_simple_merge_omp::TestTaskOpenMP::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}
bool smirnov_i_radix_sort_simple_merge_omp::TestTaskOpenMP::RunImpl() {
  std::deque<std::vector<int>> firstdq;
  std::deque<std::vector<int>> seconddq;
  bool flag = false;
#pragma omp parallel shared(flag)
  {
    int num = omp_get_thread_num();
    int all = omp_get_num_threads();
    std::vector<int> local_mas;
    int start = static_cast<int>(num * mas_.size() / all);
    int end = static_cast<int>(std::min((num + 1) * mas_.size() / all, mas_.size()));
    for (int i = start; i < end; i++) {
      local_mas.push_back(mas_[i]);
    }
    RadixSort(local_mas);
#pragma omp critical
    {
      if (!local_mas.empty()) {
        firstdq.push_back(local_mas);
      }
    }
#pragma omp barrier
#pragma omp single
    { flag = static_cast<int>(firstdq.size()) != 1; }
#pragma omp barrier
    while (flag) {
      std::vector<int> mas1{};
      std::vector<int> mas2{};
      std::vector<int> merge_mas{};
#pragma omp critical
      {
        if (static_cast<int>(firstdq.size()) >= 2) {
          mas1 = std::move(firstdq.front());
          firstdq.pop_front();
          mas2 = std::move(firstdq.front());
          firstdq.pop_front();
        }
      }
      if (!mas1.empty() && !mas2.empty()) {
        merge_mas = Merge(mas1, mas2);
      }

      if (!merge_mas.empty()) {
#pragma omp critical
        seconddq.push_back(merge_mas);
      }
#pragma omp barrier
#pragma omp single
      {
        if (static_cast<int>(firstdq.size()) == 1) {
          seconddq.push_back(std::move(firstdq.front()));
          firstdq.pop_front();
        }
        std::swap(firstdq, seconddq);
        flag = static_cast<int>(firstdq.size()) != 1;
      }
#pragma omp barrier
    }
#pragma omp barrier
#pragma omp single
    { output_ = std::move(firstdq.front()); }
  }
  return true;
}
bool smirnov_i_radix_sort_simple_merge_omp::TestTaskOpenMP::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}