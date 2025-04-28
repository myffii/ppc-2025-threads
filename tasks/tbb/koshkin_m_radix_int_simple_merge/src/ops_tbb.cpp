#include "../include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

namespace {
void RadixIntegerSortHomogenous(std::span<int> arr) {
  using TInt = std::decay_t<decltype(arr)>::value_type;

  const auto convert = [](TInt num, std::size_t i) {
    constexpr TInt kMask = 1 << ((sizeof(TInt) * CHAR_BIT) - 1);
    num = ((num & kMask) == 0U) ? (num | kMask) : (~num);
    return (num >> (i * 8)) & 0xFF;
  };

  std::vector<TInt> bufbuf(arr.size());
  std::span<TInt> buf{bufbuf};
  std::size_t counts[1 << 8];

  for (std::size_t pp = 0; pp < sizeof(TInt); ++pp) {
    std::ranges::fill(counts, 0);
    for (auto &num : arr) {
      ++counts[convert(num, pp)];
    }
    std::partial_sum(std::begin(counts), std::end(counts), std::begin(counts));
    for (auto &num : std::views::reverse(arr)) {
      buf[--counts[convert(num, pp)]] = num;
    }
    std::swap(arr, buf);
  }
}

std::vector<int> RadixIntegerSort(std::span<int> arr) {
  std::vector<int> res(arr.size());

  std::size_t left = 0;
  std::size_t right = arr.size();
  for (auto num : arr) {
    if (num < 0) {
      res[left++] = num;
    } else {
      res[--right] = num;
    }
  }

  // NOLINTBEGIN(bugprone-narrowing-conversions) :: std::size_t -> std::vector<T>::difference_type
  std::span<int> neg(res.begin(), res.begin() + left);
  std::span<int> pos(res.begin() + right, res.end());
  // NOLINTEND(bugprone-narrowing-conversions)

  RadixIntegerSortHomogenous(neg);
  RadixIntegerSortHomogenous(pos);

  std::ranges::reverse(neg);
  return res;
}
}  // namespace

bool koshkin_m_radix_int_simple_merge::TbbT::PreProcessingImpl() {
  const auto &[src, cnt] = std::pair(reinterpret_cast<int *>(task_data->inputs[0]), task_data->inputs_count[0]);
  if (cnt == 0) {
    return true;
  }

  in_ = {src, src + cnt};
  blocks_.resize(std::min(in_.size(), static_cast<std::size_t>(ppc::util::GetPPCNumThreads())));

  return true;
}

bool koshkin_m_radix_int_simple_merge::TbbT::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool koshkin_m_radix_int_simple_merge::TbbT::RunImpl() {
  auto blocks = blocks_;

  if (blocks.empty()) {
    return true;
  }

  const std::size_t per = in_.size() / blocks_.size();
  const std::size_t ext = in_.size() % blocks_.size();
  auto *ibeg = in_.data();

  std::vector<std::span<int>> intermediate_blocks(blocks.size());
  for (std::size_t i = 0; i < blocks.size(); i++) {
    auto *iend = ibeg + per + ((i < ext) ? 1 : 0);
    intermediate_blocks[i] = std::span{ibeg, iend};
    ibeg = iend;
  }

  oneapi::tbb::task_arena arena(static_cast<int>(blocks_.size()));
  arena.execute([&] {
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, tbb::this_task_arena::max_concurrency(), 1),
                              [&](const auto &r) {
                                for (std::size_t i = r.begin(); i < r.end(); i++) {
                                  blocks[i] = RadixIntegerSort(intermediate_blocks[i]);
                                }
                              });
  });

  for (std::size_t blcnt = blocks.size(), dx = 1; blcnt > 1; blcnt /= 2, dx *= 2) {
    const auto bsize = blocks[0].size();
    const auto after = blcnt / 2;

    const auto merge_and_subst = [&](std::vector<int> &a, std::vector<int> &b) {
      std::vector<int> c(a.size() + b.size());
      std::ranges::merge(a, b, c.begin());
      a = std::move(c);
      b.clear();
    };
    if (bsize > 64) {
      arena.execute([&] {
        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range<std::size_t>(0, after, after / tbb::this_task_arena::max_concurrency()),
            [&](const auto &r) {
              for (std::size_t g = r.begin(); g < r.end(); ++g) {
                merge_and_subst(blocks[2 * dx * g], blocks[(2 * dx * g) + dx]);
              }
            });
      });
    } else {
      for (std::size_t g = 0; g < after; ++g) {
        merge_and_subst(blocks[2 * dx * g], blocks[(2 * dx * g) + dx]);
      }
    }

    if (after % 2 != 0) {
      if (after == 1) {
        merge_and_subst(blocks.front(), blocks.back());
      } else {
        merge_and_subst(blocks[2 * dx * (after - 2)], blocks[2 * dx * (after - 1)]);
      }
    }
  }

  out_ = std::move(blocks.front());

  return true;
}

bool koshkin_m_radix_int_simple_merge::TbbT::PostProcessingImpl() {
  auto *tgt = reinterpret_cast<decltype(out_)::value_type *>(task_data->outputs[0]);
  std::ranges::copy(out_, tgt);
  return true;
}
