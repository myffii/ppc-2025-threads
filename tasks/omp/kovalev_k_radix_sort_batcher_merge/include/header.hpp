#pragma once

#include <memory>
#include <utility>

#include "core/task/include/task.hpp"

namespace kovalev_k_radix_sort_batcher_merge_omp {

class TestTaskOpenMP : public ppc::core::Task {
 private:
  long long int *mas_, *tmp_;
  unsigned int n_, n_input_, loc_lenght_;
  int effective_num_threads_;

 public:
  explicit TestTaskOpenMP(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  static bool RadixUnsigned(unsigned long long *, unsigned long long *, unsigned int);
  [[nodiscard]] bool RadixSigned(unsigned int, unsigned int) const;
  static bool Countbyte(unsigned long long *, int *, unsigned int, unsigned int);
  static bool OddEvenMerge(long long int *, long long int *, const long long int *, unsigned int, unsigned int);
  bool FinalMerge();
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
}  // namespace kovalev_k_radix_sort_batcher_merge_omp