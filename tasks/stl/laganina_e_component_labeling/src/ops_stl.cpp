#include "stl/laganina_e_component_labeling/include/ops_stl.hpp"

#include <algorithm>
#include <atomic>
#include <functional>
#include <future>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace {
int CountRootsInChunk(const std::vector<int>& parent, int start, int end) {
  int count = 0;
  for (int i = start; i < end; ++i) {
    if (parent[i] == i) {
      count++;
    }
  }
  return count;
}

void LabelRootsInChunk(std::vector<int>& labels, const std::vector<int>& parent, int start, int end, int label_start) {
  int current_label = label_start + 1;
  for (int i = start; i < end; ++i) {
    if (parent[i] == i) {
      labels[i] = current_label++;
    }
  }
}

void PropagateLabelsInChunk(std::vector<int>& binary, const std::vector<int>& labels, const std::vector<int>& parent,
                            int start, int end) {
  for (int i = start; i < end; ++i) {
    binary[i] = (parent[i] != -1) ? labels[parent[i]] : 0;
  }
}
}  // namespace

bool laganina_e_component_labeling_stl::TestTaskSTL::ValidationImpl() {
  if (task_data == nullptr || task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }

  const auto size = static_cast<int>(task_data->inputs_count[0] * task_data->inputs_count[1]);
  const int* input = reinterpret_cast<const int*>(task_data->inputs[0]);

  for (int i = 0; i < size; ++i) {
    if (input[i] != 0 && input[i] != 1) {
      return false;
    }
  }
  return true;
}

bool laganina_e_component_labeling_stl::TestTaskSTL::PreProcessingImpl() {
  m_ = static_cast<int>(task_data->inputs_count[0]);
  n_ = static_cast<int>(task_data->inputs_count[1]);
  binary_.resize(m_ * n_);
  const int* input = reinterpret_cast<const int*>(task_data->inputs[0]);
  std::copy_n(input, m_ * n_, binary_.begin());
  return true;
}

bool laganina_e_component_labeling_stl::TestTaskSTL::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(binary_.cbegin(), binary_.cend(), output);
  return true;
}

void laganina_e_component_labeling_stl::TestTaskSTL::InitializeParents(std::vector<int>& parent) {
  const int size = m_ * n_;
  int num_threads = std::min(ppc::util::GetPPCNumThreads(), static_cast<int>(std::thread::hardware_concurrency()));
  const int chunk_size = (size + num_threads - 1) / num_threads;

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    const int start = t * chunk_size;
    const int end = std::min(start + chunk_size, size);

    threads.emplace_back([&, start, end] {
      for (int i = start; i < end; ++i) {
        parent[i] = binary_[i] ? i : -1;
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
}

void laganina_e_component_labeling_stl::TestTaskSTL::ProcessSweep(bool reverse, std::vector<int>& parent,
                                                                  bool& changed) const {
  int num_threads = std::min(ppc::util::GetPPCNumThreads(), static_cast<int>(std::thread::hardware_concurrency()));
  const int chunk_size = (m_ + num_threads - 1) / num_threads;
  std::atomic<bool> global_changed(false);

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    const int start = t * chunk_size;
    const int end = std::min(start + chunk_size, m_);

    threads.emplace_back([&, start, end] {
      bool local_changed = false;
      for (int row_idx = start; row_idx < end; ++row_idx) {
        local_changed |= ProcessRow(row_idx, reverse, parent);
      }
      if (local_changed) {
        global_changed = true;
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
  changed = global_changed.load();
}

bool laganina_e_component_labeling_stl::TestTaskSTL::ProcessRow(int row_idx, bool reverse,
                                                                std::vector<int>& parent) const {
  const int row = reverse ? m_ - 1 - row_idx : row_idx;
  bool row_changed = false;

  for (int col_idx = 0; col_idx < n_; ++col_idx) {
    const int col = reverse ? n_ - 1 - col_idx : col_idx;
    const int current = (row * n_) + col;

    if (parent[current] == -1) {
      continue;
    }

    const int vert_neighbor_row = row - (reverse ? -1 : 1);
    const int vert_neighbor = (vert_neighbor_row * n_) + col;
    if (vert_neighbor_row >= 0 && vert_neighbor_row < m_ && parent[vert_neighbor] != -1) {
      row_changed |= UnionNodes(current, vert_neighbor, parent);
    }

    const int horz_neighbor_col = col - (reverse ? -1 : 1);
    const int horz_neighbor = (row * n_) + horz_neighbor_col;
    if (horz_neighbor_col >= 0 && horz_neighbor_col < n_ && parent[horz_neighbor] != -1) {
      row_changed |= UnionNodes(current, horz_neighbor, parent);
    }
  }

  return row_changed;
}

bool laganina_e_component_labeling_stl::TestTaskSTL::CheckNeighbor(int nr, int nc, int current,
                                                                   std::vector<int>& parent) const {
  if (nr >= 0 && nr < m_ && nc >= 0 && nc < n_) {
    const int neighbor = (nr * n_) + nc;
    if (parent[neighbor] != -1) {
      return UnionNodes(current, neighbor, parent);
    }
  }
  return false;
}

int laganina_e_component_labeling_stl::TestTaskSTL::FindRoot(std::vector<int>& parent, int x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

bool laganina_e_component_labeling_stl::TestTaskSTL::UnionNodes(int a, int b, std::vector<int>& parent) {
  int root_a = FindRoot(parent, a);
  int root_b = FindRoot(parent, b);

  if (root_a != root_b) {
    if (root_b < root_a) {
      std::swap(root_a, root_b);
    }

    parent[root_b] = root_a;

    return true;
  }
  return false;
}

void laganina_e_component_labeling_stl::TestTaskSTL::FinalizeRoots(std::vector<int>& parent) const {
  const int size = m_ * n_;
  int num_threads = std::min(ppc::util::GetPPCNumThreads(), static_cast<int>(std::thread::hardware_concurrency()));
  const int chunk_size = (size + num_threads - 1) / num_threads;

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    const int start = t * chunk_size;
    const int end = std::min(start + chunk_size, size);

    threads.emplace_back([&, start, end] {
      for (int i = start; i < end; ++i) {
        if (parent[i] != -1) {
          parent[i] = FindRoot(parent, i);
        }
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
}

void laganina_e_component_labeling_stl::TestTaskSTL::AssignLabels(std::vector<int>& parent) {
  const int size = m_ * n_;
  int num_threads = std::min(ppc::util::GetPPCNumThreads(), static_cast<int>(std::thread::hardware_concurrency()));
  std::vector<int> labels(size + 1, 0);
  const int chunk_size = (size + num_threads - 1) / num_threads;

  std::vector<int> root_counts(num_threads + 1, 0);
  {
    std::vector<std::future<int>> count_futures;
    for (int t = 0; t < num_threads; ++t) {
      const int start = t * chunk_size;
      const int end = std::min(start + chunk_size, size);
      count_futures.push_back(std::async(std::launch::async, CountRootsInChunk, std::cref(parent), start, end));
    }

    for (int t = 0; t < num_threads; ++t) {
      root_counts[t + 1] = count_futures[t].get();
    }

    for (int t = 1; t <= num_threads; ++t) {
      root_counts[t] += root_counts[t - 1];
    }
  }

  {
    std::vector<std::future<void>> label_futures;
    for (int t = 0; t < num_threads; ++t) {
      const int start = t * chunk_size;
      const int end = std::min(start + chunk_size, size);
      const int label_start = root_counts[t];
      label_futures.push_back(std::async(std::launch::async, LabelRootsInChunk, std::ref(labels), std::cref(parent),
                                         start, end, label_start));
    }

    for (auto& f : label_futures) {
      f.wait();
    }
  }

  {
    std::vector<std::future<void>> propagate_futures;
    for (int t = 0; t < num_threads; ++t) {
      const int start = t * chunk_size;
      const int end = std::min(start + chunk_size, size);
      propagate_futures.push_back(std::async(std::launch::async, PropagateLabelsInChunk, std::ref(binary_),
                                             std::cref(labels), std::cref(parent), start, end));
    }

    for (auto& f : propagate_futures) {
      f.wait();
    }
  }
}

void laganina_e_component_labeling_stl::TestTaskSTL::LabelConnectedComponents() {
  std::vector<int> parent(m_ * n_);
  InitializeParents(parent);

  constexpr int kMaxIterations = 100;
  bool changed = false;
  int iterations = 0;

  do {
    changed = false;
    ProcessSweep(false, parent, changed);
    ProcessSweep(true, parent, changed);
  } while (changed && ++iterations < kMaxIterations);

  FinalizeRoots(parent);
  AssignLabels(parent);
}

bool laganina_e_component_labeling_stl::TestTaskSTL::RunImpl() {
  LabelConnectedComponents();
  return true;
}