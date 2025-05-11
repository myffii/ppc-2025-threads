#include "stl/kondratev_ya_ccs_complex_multiplication/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

bool kondratev_ya_ccs_complex_multiplication_stl::IsZero(const std::complex<double> &value) {
  return std::norm(value) < kEpsilonForZero;
}

bool kondratev_ya_ccs_complex_multiplication_stl::IsEqual(const std::complex<double> &a,
                                                          const std::complex<double> &b) {
  return std::norm(a - b) <= kEpsilonForZero;
}

bool kondratev_ya_ccs_complex_multiplication_stl::TestTaskSTL::PreProcessingImpl() {
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

bool kondratev_ya_ccs_complex_multiplication_stl::TestTaskSTL::ValidationImpl() {
  return task_data->inputs_count[0] == 2 && task_data->outputs_count[0] == 1 && task_data->inputs[0] != nullptr &&
         task_data->inputs[1] != nullptr && task_data->outputs[0] != nullptr;
}

bool kondratev_ya_ccs_complex_multiplication_stl::TestTaskSTL::RunImpl() {
  c_ = a_ * b_;
  return true;
}

bool kondratev_ya_ccs_complex_multiplication_stl::TestTaskSTL::PostProcessingImpl() {
  *reinterpret_cast<CCSMatrix *>(task_data->outputs[0]) = c_;
  return true;
}

kondratev_ya_ccs_complex_multiplication_stl::CCSMatrix
kondratev_ya_ccs_complex_multiplication_stl::CCSMatrix::operator*(const CCSMatrix &other) const {
  CCSMatrix result({rows, other.cols});
  result.values.reserve(std::min(rows * other.cols, static_cast<int>(values.size() * other.values.size())));
  result.row_index.reserve(result.values.capacity());
  result.col_ptrs.resize(other.cols + 1, 0);

  auto temp_cols = ComputeColumns(other);
  FillResultFromTempCols(temp_cols, other.cols, result);

  return result;
}

std::vector<std::vector<std::pair<int, std::complex<double>>>>
kondratev_ya_ccs_complex_multiplication_stl::CCSMatrix::ComputeColumns(const CCSMatrix &other) const {
  int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::vector<std::pair<int, std::complex<double>>>> temp_cols(other.cols);

  int chunk_size = (other.cols + num_threads - 1) / num_threads;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    int start_col = thread_id * chunk_size;
    int end_col = std::min(start_col + chunk_size, other.cols);

    if (start_col >= other.cols) {
      break;
    }
    threads.emplace_back([&, start_col, end_col]() {
      std::vector<std::complex<double>> local_temp_col(rows);

      for (int result_col = start_col; result_col < end_col; ++result_col) {
        std::ranges::fill(local_temp_col, std::complex<double>(0.0, 0.0));

        temp_cols[result_col].reserve(std::min(rows, other.col_ptrs[result_col + 1] - other.col_ptrs[result_col]));

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
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  return temp_cols;
}

void kondratev_ya_ccs_complex_multiplication_stl::CCSMatrix::FillResultFromTempCols(
    const std::vector<std::vector<std::pair<int, std::complex<double>>>> &temp_cols, int cols, CCSMatrix &result) {
  std::vector<int> col_sizes(cols);
  for (int i = 0; i < cols; ++i) {
    col_sizes[i] = static_cast<int>(temp_cols[i].size());
  }

  result.col_ptrs[0] = 0;
  for (int i = 0; i < cols; ++i) {
    result.col_ptrs[i + 1] = result.col_ptrs[i] + col_sizes[i];
  }

  int total_nonzeros = result.col_ptrs[cols];
  result.values.resize(total_nonzeros);
  result.row_index.resize(total_nonzeros);

  int num_threads = ppc::util::GetPPCNumThreads();
  int chunk_size = (cols + num_threads - 1) / num_threads;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    int start_col = thread_id * chunk_size;
    int end_col = std::min(start_col + chunk_size, cols);

    if (start_col >= cols) {
      break;
    }

    threads.emplace_back([&, start_col, end_col]() {
      for (int col = start_col; col < end_col; ++col) {
        int offset = result.col_ptrs[col];
        const auto &col_values = temp_cols[col];

        for (size_t j = 0; j < col_values.size(); ++j) {
          result.row_index[offset + j] = col_values[j].first;
          result.values[offset + j] = col_values[j].second;
        }
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}