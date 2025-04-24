#include "tbb/shlyakov_m_shell_sort/include/ops_tbb.hpp"

#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>

#include <algorithm>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <utility>
#include <vector>

namespace shlyakov_m_shell_sort_tbb {

bool TestTaskTBB::PreProcessingImpl() {
  const std::size_t sz = task_data->inputs_count[0];
  auto* ptr = reinterpret_cast<int*>(task_data->inputs[0]);

  input_.assign(ptr, ptr + sz);
  output_ = input_;
  return true;
}

bool TestTaskTBB::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool TestTaskTBB::RunImpl() {
  const int n = static_cast<int>(input_.size());
  if (n < 2) {
    return true;
  }

  const int max_threads = ppc::util::GetPPCNumThreads();
  int threads = std::min(max_threads, n);
  const int seg_size = (n + threads - 1) / threads;

  std::vector<std::pair<int, int>> segs;
  segs.reserve(threads);
  for (int idx = 0; idx < threads; ++idx) {
    const int l = idx * seg_size;
    const int r = std::min(n - 1, l + seg_size - 1);
    segs.emplace_back(l, r);
  }

  tbb::task_arena arena(threads);
  arena.execute([&] {
    tbb::task_group tg;
    for (const auto& seg : segs) {
      const int l = seg.first;
      const int r = seg.second;
      tg.run([this, l, r] { ShellSort(l, r, input_); });
    }
    tg.wait();
  });

  std::vector<int> buf;
  int end = segs.front().second;
  for (std::size_t i = 1; i < segs.size(); ++i) {
    const int r = segs[i].second;
    Merge(0, end, r, input_, buf);
    end = r;
  }

  output_ = input_;
  return true;
}

void ShellSort(int left, int right, std::vector<int>& arr) {
  int gap = 1;
  const int size = right - left + 1;
  while (gap <= size / 3) {
    gap = gap * 3 + 1;
  }

  for (; gap > 0; gap /= 3) {
    for (int k = left + gap; k <= right; ++k) {
      const int val = arr[k];
      int j = k;
      while (j >= left + gap && arr[j - gap] > val) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = val;
    }
  }
}

void Merge(int left, int mid, int right, std::vector<int>& arr, std::vector<int>& buffer) {
  const int merge_size = right - left + 1;
  if (buffer.size() < static_cast<std::size_t>(merge_size)) {
    buffer.resize(static_cast<std::size_t>(merge_size));
  }

  int i = left;
  int j = mid + 1;
  int k = 0;

  while (i <= mid && j <= right) {
    buffer[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
  }
  while (i <= mid) {
    buffer[k++] = arr[i++];
  }
  while (j <= right) {
    buffer[k++] = arr[j++];
  }

  for (int idx = 0; idx < merge_size; ++idx) {
    arr[left + idx] = buffer[idx];
  }
}

bool TestTaskTBB::PostProcessingImpl() {
  for (std::size_t idx = 0; idx < output_.size(); ++idx) {
    reinterpret_cast<int*>(task_data->outputs[0])[idx] = output_[idx];
  }
  return true;
}

}  // namespace shlyakov_m_shell_sort_tbb