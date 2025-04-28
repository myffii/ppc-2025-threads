#include "tbb/solovyev_d_shell_sort_simple/include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_for.h>

#include <cmath>
#include <cstddef>
#include <vector>

bool solovyev_d_shell_sort_simple_tbb::TaskTBB::PreProcessingImpl() {
  size_t input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  return true;
}

bool solovyev_d_shell_sort_simple_tbb::TaskTBB::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool solovyev_d_shell_sort_simple_tbb::TaskTBB::RunImpl() {
  for (int gap = static_cast<int>(input_.size()) / 2; gap > 0; gap /= 2) {
    tbb::parallel_for(0, gap, [this, gap](int i) {
      for (size_t f = gap + i; f < input_.size(); f += gap) {
        int val = input_[f];
        int j = static_cast<int>(f);
        while (j >= gap && input_[j - gap] > val) {
          input_[j] = input_[j - gap];
          j -= gap;
        }
        input_[j] = val;
      }
    });
  }
  return true;
}

bool solovyev_d_shell_sort_simple_tbb::TaskTBB::PostProcessingImpl() {
  for (size_t i = 0; i < input_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = input_[i];
  }
  return true;
}
