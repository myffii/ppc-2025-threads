#include "tbb/smirnov_i_radix_sort_simple_merge/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <numeric>
#include <utility>
#include <vector>

#include "oneapi/tbb/mutex.h"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

std::vector<int> smirnov_i_radix_sort_simple_merge_tbb::TestTaskTBB::Merge(std::vector<int>& mas1,
                                                                           std::vector<int>& mas2) {
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
void smirnov_i_radix_sort_simple_merge_tbb::TestTaskTBB::RadixSort(std::vector<int>& mas) {
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
void smirnov_i_radix_sort_simple_merge_tbb::TestTaskTBB::SortChunk(int i, int size, int nth, int& start,
                                                                   tbb::mutex& mtx_start, tbb::mutex& mtx_firstdq,
                                                                   std::deque<std::vector<int>>& firstdq) {
  if (size <= 0 || mas_.empty()) {
    return;
  }
  int self_offset = size / nth;
  if (size % nth != 0) {
    self_offset += static_cast<int>(i < size % nth);
  }
  std::vector<int> local_mas(self_offset);
  mtx_start.lock();
  int self_start = start;
  start += self_offset;
  mtx_start.unlock();
  if (self_start < 0 || self_start >= static_cast<int>(mas_.size()) || self_offset < 0 ||
      self_start + self_offset > static_cast<int>(mas_.size())) {
    return;
  }
  std::copy(mas_.begin() + self_start, mas_.begin() + self_start + self_offset, local_mas.begin());
  if (!local_mas.empty()) {
    RadixSort(local_mas);
    mtx_firstdq.lock();
    firstdq.push_back(std::move(local_mas));
    mtx_firstdq.unlock();
  }
}
bool smirnov_i_radix_sort_simple_merge_tbb::TestTaskTBB::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  mas_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);
  return true;
}
bool smirnov_i_radix_sort_simple_merge_tbb::TestTaskTBB::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}
bool smirnov_i_radix_sort_simple_merge_tbb::TestTaskTBB::RunImpl() {
  std::deque<std::vector<int>> firstdq;
  std::deque<std::vector<int>> seconddq;
  tbb::task_group tg;
  int size = static_cast<int>(mas_.size());
  const int nth = std::min(size, tbb::this_task_arena::max_concurrency());
  tbb::mutex mtx;
  tbb::mutex mtx_firstdq;
  tbb::mutex mtx_start;
  int start = 0;

  for (int i = 0; i < nth; i++) {
    tg.run([this, i, size, nth, &start, &mtx_firstdq, &firstdq, &mtx_start]() {
      SortChunk(i, size, nth, start, mtx_start, mtx_firstdq, firstdq);
    });
  }
  tg.wait();
  bool flag = static_cast<int>(firstdq.size()) != 1;
  while (flag) {
    for (int i = 0; i < nth; i++) {
      tg.run([&firstdq, &mtx, &seconddq]() {
        std::vector<int> mas1{};
        std::vector<int> mas2{};
        std::vector<int> merge_mas{};
        mtx.lock();
        if (static_cast<int>(firstdq.size()) >= 2) {
          mas1 = std::move(firstdq.front());
          firstdq.pop_front();
          mas2 = std::move(firstdq.front());
          firstdq.pop_front();
        } else {
          mtx.unlock();
          return;
        }
        mtx.unlock();
        if (!mas1.empty() && !mas2.empty()) {
          merge_mas = Merge(mas1, mas2);
        }
        if (!merge_mas.empty()) {
          mtx.lock();
          seconddq.push_back(std::move(merge_mas));
          mtx.unlock();
        }
      });
    }
    tg.wait();
    if (static_cast<int>(firstdq.size()) == 1) {
      seconddq.push_back(std::move(firstdq.front()));
      firstdq.pop_front();
    }
    std::swap(firstdq, seconddq);
    flag = static_cast<int>(firstdq.size()) != 1;
  }
  output_ = std::move(firstdq.front());
  return true;
}
bool smirnov_i_radix_sort_simple_merge_tbb::TestTaskTBB::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}