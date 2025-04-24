#ifndef OPS_OMP_HPP
#define OPS_OMP_HPP

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using Bigint = long long;
using namespace std;

namespace belov_a_radix_batcher_mergesort_omp {

class RadixBatcherMergesortParallel : public ppc::core::Task {
 public:
  explicit RadixBatcherMergesortParallel(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void Sort(std::span<Bigint> arr);

 private:
  vector<Bigint> array_;  // input unsorted numbers array
  size_t n_ = 0;          // array size

  static void RadixSort(vector<Bigint>& arr, bool invert);
  static void CountingSort(vector<Bigint>& arr, Bigint digit_place);
  static int GetNumberDigitCapacity(Bigint num);

  static void SortParallel(vector<Bigint>& arr);
  static void BatcherMergeParallel(vector<Bigint>& arr, int num_threads);
};

}  // namespace belov_a_radix_batcher_mergesort_omp

#endif  // OPS_OMP_HPP