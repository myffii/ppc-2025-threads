#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

namespace tyshkevich_a_hoare_simple_merge_omp {

template <typename T>
struct ArrayPiece {
  T* first;
  T* last;

  [[nodiscard]] std::size_t Size() const noexcept { return last - first; }

  operator std::span<T>() const noexcept { return std::span<T>(first, last); }

  void Concat(const ArrayPiece& other) { last = other.last; }

  static std::vector<ArrayPiece> Partition(std::span<T> arr, std::size_t pieces) {
    const std::size_t delta = arr.size() / pieces;
    const std::size_t extra = arr.size() % pieces;

    std::vector<ArrayPiece> v(pieces);
    auto* cur = arr.data();
    for (std::size_t i = 0; i < pieces; i++) {
      const std::size_t sz = delta + ((i < extra) ? 1 : 0);
      v[i] = {.first = cur, .last = cur + sz};
      cur += sz;
    }

    return v;
  }
};

template <typename T, typename Comparator>
class HoareSortTask : public ppc::core::Task {
 public:
  explicit HoareSortTask(ppc::core::TaskDataPtr task_data, Comparator cmp) : Task(std::move(task_data)), cmp_(cmp) {}

  bool ValidationImpl() override { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

  bool PreProcessingImpl() override {
    input_ = {reinterpret_cast<const T*>(task_data->inputs[0]), task_data->inputs_count[0]};
    output_ = {reinterpret_cast<T*>(task_data->outputs[0]), task_data->outputs_count[0]};
    return true;
  }

  bool RunImpl() override {
    std::copy(input_.begin(), input_.end(), output_.begin());

    const std::size_t concurrency = std::min(output_.size(), std::size_t(ppc::util::GetPPCNumThreads()));
    if (concurrency == 0) {
      return true;
    }

    auto pieces = ArrayPiece<T>::Partition(output_, concurrency);

#pragma omp parallel for
    for (int tnum = 0; tnum < int(concurrency); tnum++) {
      auto& piece = pieces[tnum];
      HoareSort(piece, 0, piece.Size() - 1);
    }

    std::size_t leap = 1;
    for (std::size_t reduc = concurrency; reduc > 1; reduc /= 2) {
      const auto dl = 2 * leap;
      const auto nextreduc = reduc / 2;

#pragma omp parallel for if (pieces[0].Size() > kParallelizeThreshold)
      for (int g = 0; g < static_cast<int>(nextreduc); ++g) {
        auto& head = pieces[dl * g];
        auto& tail = pieces[(dl * g) + leap];
        std::inplace_merge(head.first, tail.first, tail.last, cmp_);
        head.Concat(tail);
      }

      if (nextreduc % 2 != 0) {
        ArrayPiece<T>* head = nullptr;
        ArrayPiece<T>* tail = nullptr;
        if (nextreduc == 1) {
          head = &pieces.front();
          tail = &pieces.back();
        } else {
          head = &pieces[dl * (nextreduc - 2)];
          tail = &pieces[dl * (nextreduc - 1)];
        }
        std::inplace_merge(head->first, tail->first, tail->last, cmp_);
        head->Concat(*tail);
      }

      leap *= 2;
    }

    return true;
  }

  bool PostProcessingImpl() override {
    // output_ is being modified directly during Run
    return true;
  }

 private:
  void HoareSort(std::span<T> arr, int64_t low, int64_t high) {
    const auto partition = [&cmp = this->cmp_](std::span<T> region, int64_t plo, int64_t phi) -> int64_t {
      const auto& pivot = region[phi];
      int64_t e = plo - 1;
      for (int64_t j = plo; j <= phi - 1; j++) {
        if (cmp(region[j], pivot)) {
          std::swap(region[++e], region[j]);
        }
      }
      std::swap(region[e + 1], region[phi]);
      return e + 1;
    };

    if (low < high) {
      int64_t p = partition(arr, low, high);
      HoareSort(arr, low, p - 1);
      HoareSort(arr, p + 1, high);
    }
  };

  Comparator cmp_;

  std::span<const T> input_;
  std::span<T> output_;

  static constexpr std::size_t kParallelizeThreshold = 32;
};

template <typename T, typename Comparator>
HoareSortTask<T, Comparator> CreateHoareTestTask(ppc::core::TaskDataPtr task_data, Comparator cmp) {
  return HoareSortTask<T, Comparator>(std::move(task_data), cmp);
}

}  // namespace tyshkevich_a_hoare_simple_merge_omp