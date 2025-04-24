#include "tbb/kondratev_ya_ccs_complex_multiplication/include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <utility>
#include <vector>

bool kondratev_ya_ccs_complex_multiplication_tbb::IsZero(const std::complex<double> &value) {
  return std::norm(value) < kEpsilonForZero;
}

bool kondratev_ya_ccs_complex_multiplication_tbb::IsEqual(const std::complex<double> &a,
                                                          const std::complex<double> &b) {
  return std::norm(a - b) <= kEpsilonForZero;
}

bool kondratev_ya_ccs_complex_multiplication_tbb::TestTaskTBB::PreProcessingImpl() {
  a_ = *reinterpret_cast<CCSMatrix *>(task_data->inputs[0]);
  b_ = *reinterpret_cast<CCSMatrix *>(task_data->inputs[1]);

  if (a_.rows == 0 || a_.cols == 0 || b_.rows == 0 || b_.cols == 0) {
    return false;
  }

  if (a_.cols != b_.rows) {
    return false;
  }

  return true;
}

bool kondratev_ya_ccs_complex_multiplication_tbb::TestTaskTBB::ValidationImpl() {
  return task_data->inputs_count[0] == 2 && task_data->outputs_count[0] == 1 && task_data->inputs[0] != nullptr &&
         task_data->inputs[1] != nullptr && task_data->outputs[0] != nullptr;
}

bool kondratev_ya_ccs_complex_multiplication_tbb::TestTaskTBB::RunImpl() {
  c_ = a_ * b_;
  return true;
}

bool kondratev_ya_ccs_complex_multiplication_tbb::TestTaskTBB::PostProcessingImpl() {
  *reinterpret_cast<CCSMatrix *>(task_data->outputs[0]) = c_;
  return true;
}

kondratev_ya_ccs_complex_multiplication_tbb::CCSMatrix
kondratev_ya_ccs_complex_multiplication_tbb::CCSMatrix::operator*(const CCSMatrix &other) const {
  CCSMatrix result({rows, other.cols});
  result.values.reserve(std::min(rows * other.cols, static_cast<int>(values.size() * other.values.size())));
  result.row_index.reserve(result.values.capacity());
  result.col_ptrs.resize(other.cols + 1, 0);

  std::vector<std::vector<std::pair<int, std::complex<double>>>> temp_cols(other.cols);

  tbb::parallel_for(0, other.cols, [&](int result_col) {
    std::vector<std::complex<double>> local_temp_col(rows, std::complex<double>(0.0, 0.0));

    for (int k = other.col_ptrs[result_col]; k < other.col_ptrs[result_col + 1]; k++) {
      int row_other = other.row_index[k];
      std::complex<double> val_other = other.values[k];

      for (int i = col_ptrs[row_other]; i < col_ptrs[row_other + 1]; i++) {
        int row_this = row_index[i];
        local_temp_col[row_this] += values[i] * val_other;
      }
    }

    for (int i = 0; i < rows; i++) {
      if (!IsZero(local_temp_col[i])) {
        temp_cols[result_col].emplace_back(i, local_temp_col[i]);
      }
    }
  });

  std::vector<int> col_sizes(other.cols);
  std::ranges::transform(temp_cols, col_sizes.begin(), [](const auto &col) { return static_cast<int>(col.size()); });

  std::vector<int> col_offsets(other.cols + 1, 0);
  std::partial_sum(col_sizes.begin(), col_sizes.end(), col_offsets.begin() + 1);

  int total_nonzero = col_offsets[other.cols];
  result.values.resize(total_nonzero);
  result.row_index.resize(total_nonzero);

  for (int i = 0; i <= other.cols; i++) {
    result.col_ptrs[i] = col_offsets[i];
  }

  tbb::parallel_for(0, other.cols, [&](int col) {
    int offset = col_offsets[col];

    for (const auto &[row, value] : temp_cols[col]) {
      result.row_index[offset] = row;
      result.values[offset] = value;
      offset++;
    }
  });

  return result;
}