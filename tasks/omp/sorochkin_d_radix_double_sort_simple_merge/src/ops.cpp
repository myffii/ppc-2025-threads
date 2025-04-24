#include "../include/ops.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#include "core/util/include/util.hpp"

namespace {
template <typename T>
constexpr size_t Bytes() {
  return sizeof(T);
}
template <typename T>
constexpr size_t Bits() {
  return Bytes<T>() * CHAR_BIT;
}
class Bitutil {
 private:
  union du64 {
    double d;
    uint64_t u;
    static constexpr uint64_t kMask = 1ULL << ((sizeof(uint64_t) * CHAR_BIT) - 1);
  };

 public:
  static constexpr uint64_t AsU64(double x) {
    const du64 r{.d = x};
    return ((r.u & du64::kMask) != 0U) ? ~r.u : r.u | du64::kMask;
  }

  template <typename T>
    requires std::is_floating_point_v<T> or std::is_integral_v<T>
  static constexpr uint8_t ByteAt(const T &val, uint8_t idx) {
    return (val >> (idx * 8)) & 0xFF;
  }
};

void RadixSort(std::span<double> v) {
  constexpr size_t kBase = 1 << CHAR_BIT;

  std::vector<double> aux_buf(v.size());
  std::span<double> aux{aux_buf};

  std::array<std::size_t, kBase> count;

  for (std::size_t ib = 0; ib < Bytes<double>(); ++ib) {
    std::ranges::fill(count, 0);
    std::ranges::for_each(v, [&](auto el) { ++count[Bitutil::ByteAt(Bitutil::AsU64(el), ib)]; });
    std::partial_sum(count.begin(), count.end(), count.begin());
    std::ranges::for_each(std::ranges::reverse_view(v),
                          [&](auto el) { aux[--count[Bitutil::ByteAt(Bitutil::AsU64(el), ib)]] = el; });
    std::swap(v, aux);
  }
}
}  // namespace

bool sorochkin_d_radix_double_sort_simple_merge_omp::SortTask::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool sorochkin_d_radix_double_sort_simple_merge_omp::SortTask::PreProcessingImpl() {
  std::span<double> src = {reinterpret_cast<double *>(task_data->inputs[0]), task_data->inputs_count[0]};
  input_.assign(src.begin(), src.end());
  output_.reserve(input_.size());
  return true;
}

bool sorochkin_d_radix_double_sort_simple_merge_omp::SortTask::RunImpl() {
  output_ = input_;

  const auto size = input_.size();
  if (size == 0) {
    return true;
  }
  const auto numthreads = std::min<std::size_t>(size, ppc::util::GetPPCNumThreads());

  std::vector<std::size_t> distrib(numthreads, size / numthreads);
  distrib[numthreads - 1] += size % numthreads;
  std::vector<std::size_t> offsets(numthreads);
  if (numthreads > 1) {
    std::partial_sum(distrib.begin(), distrib.end() - 1, offsets.begin() + 1);
  }

  std::vector<std::span<double>> chunks(numthreads);
  std::ranges::generate(chunks, [&, i = 0]() mutable {
    const auto j = i++;
    return std::span{output_}.subspan(offsets[j], distrib[j]);
  });

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(numthreads); i++) {
    RadixSort(chunks[i]);
  }

  for (std::size_t i = 1, j = numthreads; j > 1; i *= 2, j /= 2) {
    const auto multithreaded = chunks.front().size() > 48;
    const auto jh = j / 2;
    const auto merge = [&](std::span<double> &master, std::span<double> &slave) {
      std::inplace_merge(master.begin(), slave.begin(), slave.end());
      master = std::span{master.begin(), slave.end()};
    };
#pragma omp parallel for if (multithreaded)
    for (int k = 0; k < static_cast<int>(jh); ++k) {
      const auto idx = 2 * i * k;
      merge(chunks[idx], chunks[idx + i]);
    }
    if (jh == 1) {
      merge(chunks.front(), chunks.back());
    } else if (jh % 2 != 0) {
      merge(chunks[2 * i * (jh - 2)], chunks[2 * i * (jh - 1)]);
    }
  }

  return true;
}

bool sorochkin_d_radix_double_sort_simple_merge_omp::SortTask::PostProcessingImpl() {
  std::ranges::copy(output_, reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
