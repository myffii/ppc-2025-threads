#include "../include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

namespace {
auto Translate(double e, std::size_t i) {
  const uint64_t mask = 1ULL << ((sizeof(uint64_t) * 8) - 1);
  const union {
    double dbl;
    uint64_t uint64;
  } uni{.dbl = e};
  const uint64_t u = ((uni.uint64 & mask) == 0U) ? (uni.uint64 | mask) : (~uni.uint64);
  return (u >> (i * 8)) & 0xFF;
}

void Radix(double *p, int len) {
  std::vector<double> tmpb(len);
  double *tmp = tmpb.data();
  std::vector<std::size_t> cnt(1 << 8, 0);

  for (std::size_t i = 0; i < sizeof(double); ++i) {
    for (auto &c : cnt) {
      c = 0;
    }
    for (int j = 0; j < len; j++) {
      ++cnt[Translate(p[j], i)];
    }
    for (std::size_t j = 1; j < cnt.size(); ++j) {
      cnt[j] += cnt[j - 1];
    }
    for (int j = len; j-- > 0;) {
      auto &e = p[j];
      const auto nidx = --cnt[Translate(e, i)];
      tmp[nidx] = e;
    }
    std::swap(p, tmp);
  }
}

void OddEvenBatcherMergeBlocksStep(std::pair<double *, int> &left, std::pair<double *, int> &right) {
  std::inplace_merge(left.first, right.first, right.first + right.second);
  left.second += right.second;
}

void ParallelOddEvenBatcherMerge(int bsz, std::vector<std::pair<double *, int>> &vb, int par_if_greater) {
  for (int step = 1, pack = int(vb.size()); pack > 1; step *= 2, pack /= 2) {
#pragma omp parallel for if ((bsz / step) > par_if_greater)
    for (int off = 0; off < pack / 2; ++off) {
      OddEvenBatcherMergeBlocksStep(vb[2 * step * off], vb[(2 * step * off) + step]);
    }
    if ((pack / 2) - 1 == 0) {
      OddEvenBatcherMergeBlocksStep(vb.front(), vb.back());
    } else if ((pack / 2) % 2 != 0) {
      OddEvenBatcherMergeBlocksStep(vb[2 * step * ((pack / 2) - 2)], vb[2 * step * ((pack / 2) - 1)]);
    }
  }
}
}  // namespace

bool petrov_a_radix_double_batcher_omp::TestTaskParallelOmp::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool petrov_a_radix_double_batcher_omp::TestTaskParallelOmp::PreProcessingImpl() {
  in_.assign(reinterpret_cast<double *>(task_data->inputs[0]),
             reinterpret_cast<double *>(task_data->inputs[0]) + task_data->inputs_count[0]);
  return true;
}

bool petrov_a_radix_double_batcher_omp::TestTaskParallelOmp::RunImpl() {
  const int sz = int(in_.size());
  const int thr = std::min(sz, ppc::util::GetPPCNumThreads());
  if (0 == sz) {
    return true;
  }

  res_.resize(in_.size());
  std::ranges::copy(in_, res_.begin());

  const int bsz = sz / thr;
  const int bex = sz % thr;

  std::vector<std::pair<double *, int>> vb(thr);
  for (int i = 0; i < thr; i++) {
    vb[i] = std::make_pair(res_.data() + (i * bsz), bsz);  // NOLINT(*narrow*)
  }
  vb[vb.size() - 1].second += bex;

#pragma omp parallel for
  for (int i = 0; i < thr; i++) {
    const auto &[p, l] = vb[i];
    Radix(p, l);
  }

  ParallelOddEvenBatcherMerge(bsz, vb, 33);

  return true;
}

bool petrov_a_radix_double_batcher_omp::TestTaskParallelOmp::PostProcessingImpl() {
  std::ranges::copy(res_, reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
