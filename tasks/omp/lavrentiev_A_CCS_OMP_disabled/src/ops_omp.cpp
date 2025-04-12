#include "omp/lavrentiev_A_CCS_OMP/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

lavrentiev_a_ccs_omp::Sparse lavrentiev_a_ccs_omp::CCSOMP::ConvertToSparse(std::pair<int, int> size,
                                                                           const std::vector<double> &values) {
  auto [nsize, elements, rows, columns_sum] = Sparse();
  columns_sum.resize(size.second);
  for (int i = 0; i < size.second; ++i) {
    for (int j = 0; j < size.first; ++j) {
      if (values[i + (size.second * j)] != 0) {
        elements.emplace_back(values[i + (size.second * j)]);
        rows.emplace_back(j);
        columns_sum[i] += 1;
      }
    }
    if (i != size.second - 1) {
      columns_sum[i + 1] = columns_sum[i];
    }
  }
  return {.size = size, .elements = elements, .rows = rows, .columnsSum = columns_sum};
}

lavrentiev_a_ccs_omp::Sparse lavrentiev_a_ccs_omp::CCSOMP::Transpose(const Sparse &sparse) {
  auto [size, elements, rows, columns_sum] = Sparse();
  size.first = sparse.size.second;
  size.second = sparse.size.first;
  int need_size = std::max(sparse.size.first, sparse.size.second);
  std::vector<std::vector<double>> new_elements(need_size);
  std::vector<std::vector<int>> new_indexes(need_size);
  int counter = 0;
  for (int i = 0; i < static_cast<int>(sparse.columnsSum.size()); ++i) {
    for (int j = 0; j < GetElementsCount(i, sparse.columnsSum); ++j) {
      new_elements[sparse.rows[counter]].emplace_back(sparse.elements[counter]);
      new_indexes[sparse.rows[counter]].emplace_back(i);
      counter++;
    }
  }
  for (int i = 0; i < static_cast<int>(new_elements.size()); ++i) {
    for (int j = 0; j < static_cast<int>(new_elements[i].size()); ++j) {
      elements.emplace_back(new_elements[i][j]);
      rows.emplace_back(new_indexes[i][j]);
    }
    if (i > 0) {
      columns_sum.emplace_back(new_elements[i].size() + columns_sum[i - 1]);
    } else {
      columns_sum.emplace_back(new_elements[i].size());
    }
  }
  return {.size = size, .elements = elements, .rows = rows, .columnsSum = columns_sum};
}

int lavrentiev_a_ccs_omp::CCSOMP::CalculateStartIndex(int index, const std::vector<int> &columns_sum) {
  if (index != 0) {
    return columns_sum[index] - GetElementsCount(index, columns_sum);
  }
  return 0;
}

lavrentiev_a_ccs_omp::Sparse lavrentiev_a_ccs_omp::CCSOMP::MatMul(const Sparse &matrix1, const Sparse &matrix2) {
  Sparse result_matrix;
  result_matrix.columnsSum.resize(matrix2.size.second);
  auto new_matrix1 = Transpose(matrix1);
  std::vector<std::pair<std::vector<double>, std::vector<int>>> threads_data(ppc::util::GetPPCNumThreads());
#pragma omp parallel
  {
    std::pair<std::vector<double>, std::vector<int>> current_thread_data;
#pragma omp for
    for (int i = 0; i < static_cast<int>(matrix2.columnsSum.size()); ++i) {
      for (int j = 0; j < static_cast<int>(new_matrix1.columnsSum.size()); ++j) {
        double sum = 0.0;
        for (int n = 0; n < GetElementsCount(j, new_matrix1.columnsSum); n++) {
          for (int n2 = 0; n2 < GetElementsCount(i, matrix2.columnsSum); n2++) {
            if (new_matrix1.rows[CalculateStartIndex(j, new_matrix1.columnsSum) + n] ==
                matrix2.rows[CalculateStartIndex(i, matrix2.columnsSum) + n2]) {
              sum += new_matrix1.elements[n + CalculateStartIndex(j, new_matrix1.columnsSum)] *
                     matrix2.elements[n2 + CalculateStartIndex(i, matrix2.columnsSum)];
            }
          }
        }
        if (sum != 0) {
          current_thread_data.first.push_back(sum);
          current_thread_data.second.push_back(j);
          result_matrix.columnsSum[i]++;
        }
      }
    }
    threads_data[omp_get_thread_num()] = std::move(current_thread_data);
  }
  for (size_t i = 1; i < result_matrix.columnsSum.size(); ++i) {
    result_matrix.columnsSum[i] = result_matrix.columnsSum[i] + result_matrix.columnsSum[i - 1];
  }
  if (!result_matrix.columnsSum.empty()) {
    result_matrix.elements.resize(result_matrix.columnsSum.back());
    result_matrix.rows.resize(result_matrix.columnsSum.back());
  }
  int count = 0;
  for (size_t i = 0; i < threads_data.size(); ++i) {
    std::ranges::copy(threads_data[i].first, result_matrix.elements.begin() + count);
    std::ranges::copy(threads_data[i].second, result_matrix.rows.begin() + count);
    count += static_cast<int>(threads_data[i].first.size());
  }
  result_matrix.size.first = matrix2.size.second;
  result_matrix.size.second = matrix2.size.second;
  return {.size = result_matrix.size,
          .elements = result_matrix.elements,
          .rows = result_matrix.rows,
          .columnsSum = result_matrix.columnsSum};
}

int lavrentiev_a_ccs_omp::CCSOMP::GetElementsCount(int index, const std::vector<int> &columns_sum) {
  if (index == 0) {
    return columns_sum[index];
  }
  return columns_sum[index] - columns_sum[index - 1];
}

std::vector<double> lavrentiev_a_ccs_omp::CCSOMP::ConvertFromSparse(const Sparse &matrix) {
  std::vector<double> nmatrix(matrix.size.first * matrix.size.second);
  int counter = 0;
  for (size_t i = 0; i < matrix.columnsSum.size(); ++i) {
    for (int j = 0; j < GetElementsCount(static_cast<int>(i), matrix.columnsSum); ++j) {
      nmatrix[i + (matrix.size.second * matrix.rows[counter])] = matrix.elements[counter];
      counter++;
    }
  }
  return nmatrix;
}

bool lavrentiev_a_ccs_omp::CCSOMP::PreProcessingImpl() {
  A_.size = {static_cast<int>(task_data->inputs_count[0]), static_cast<int>(task_data->inputs_count[1])};
  B_.size = {static_cast<int>(task_data->inputs_count[2]), static_cast<int>(task_data->inputs_count[3])};
  if (IsEmpty()) {
    return true;
  }
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  auto am = std::vector<double>(in_ptr, in_ptr + (A_.size.first * A_.size.second));
  A_ = ConvertToSparse(A_.size, am);
  auto *in_ptr2 = reinterpret_cast<double *>(task_data->inputs[1]);
  auto bm = std::vector<double>(in_ptr2, in_ptr2 + (B_.size.first * B_.size.second));
  B_ = ConvertToSparse(B_.size, bm);
  return true;
}

bool lavrentiev_a_ccs_omp::CCSOMP::IsEmpty() const {
  return A_.size.first * A_.size.second == 0 || B_.size.first * B_.size.second == 0;
}

bool lavrentiev_a_ccs_omp::CCSOMP::ValidationImpl() {
  return task_data->inputs_count[0] * task_data->inputs_count[3] == task_data->outputs_count[0] &&
         task_data->inputs_count[0] == task_data->inputs_count[3] &&
         task_data->inputs_count[1] == task_data->inputs_count[2];
}

bool lavrentiev_a_ccs_omp::CCSOMP::RunImpl() {
  Answer_ = MatMul(A_, B_);
  return true;
}

bool lavrentiev_a_ccs_omp::CCSOMP::PostProcessingImpl() {
  std::ranges::copy(ConvertFromSparse(Answer_), reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}