#include "stl/sadikov_I_SparseMatrixMultiplication/include/SparseMatrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

namespace sadikov_i_sparse_matrix_multiplication_task_stl {
SparseMatrix SparseMatrix::Transpose(const SparseMatrix& matrix) {
  MatrixComponents components;
  auto max_size = std::max(matrix.GetRowsCount(), matrix.GetColumnsCount());
  std::vector<std::vector<double>> intermediate_values(max_size);
  std::vector<std::vector<int>> intermediate_indexes(max_size);
  int counter = 0;
  for (size_t i = 0; i < matrix.GetElementsSum().size(); ++i) {
    auto limit = i == 0 ? matrix.GetElementsSum()[0] : matrix.GetElementsSum()[i] - matrix.GetElementsSum()[i - 1];
    for (int j = 0; j < limit; ++j) {
      intermediate_values[matrix.GetRows()[counter]].emplace_back(matrix.GetValues()[counter]);
      intermediate_indexes[matrix.GetRows()[counter]].emplace_back(i);
      counter++;
    }
  }
  for (size_t i = 0; i < intermediate_values.size(); ++i) {
    for (size_t j = 0; j < intermediate_values[i].size(); ++j) {
      components.m_values.emplace_back(intermediate_values[i][j]);
      components.m_rows.emplace_back(intermediate_indexes[i][j]);
    }
    i == 0 ? components.m_elementsSum.emplace_back(intermediate_values[i].size())
           : components.m_elementsSum.emplace_back(intermediate_values[i].size() + components.m_elementsSum[i - 1]);
  }
  return {matrix.GetColumnsCount(), matrix.GetRowsCount(), components};
}
double SparseMatrix::CalculateSum(const SparseMatrix& fmatrix, const SparseMatrix& smatrix,
                                  const std::vector<int>& felements_sum, const std::vector<int>& selements_sum,
                                  int i_index, int j_index) {
  int fmatrix_elements_count = GetElementsCount(j_index, felements_sum);
  int smatrix_elements_count = GetElementsCount(i_index, selements_sum);
  int fmatrix_start_index = j_index != 0 ? felements_sum[j_index] - fmatrix_elements_count : 0;
  int smatrix_start_index = i_index != 0 ? selements_sum[i_index] - smatrix_elements_count : 0;
  double sum = 0.0;
  for (int i = 0; i < fmatrix_elements_count; i++) {
    for (int j = 0; j < smatrix_elements_count; j++) {
      if (fmatrix.GetRows()[fmatrix_start_index + i] == smatrix.GetRows()[smatrix_start_index + j]) {
        sum += fmatrix.GetValues()[i + fmatrix_start_index] * smatrix.GetValues()[j + smatrix_start_index];
      }
    }
  }
  return sum;
}

std::vector<std::pair<int, int>> SparseMatrix::CalculateSeparation(size_t vector_size) {
  int n = static_cast<int>(vector_size) % ppc::util::GetPPCNumThreads();
  int count = static_cast<int>(vector_size) / ppc::util::GetPPCNumThreads();
  std::vector<std::pair<int, int>> threads_start_indexes(ppc::util::GetPPCNumThreads());
  for (int i = 0; i < static_cast<int>(threads_start_indexes.size()); ++i) {
    if (i == 0) {
      threads_start_indexes[i] = {0, count};
    } else {
      threads_start_indexes[i] = {threads_start_indexes[i - 1].second, threads_start_indexes[i - 1].second + count};
    }
    if (i < n) {
      threads_start_indexes[i].second++;
    }
  }
  return threads_start_indexes;
}

SparseMatrix SparseMatrix::operator*(SparseMatrix& smatrix) const {
  auto fmatrix = Transpose(*this);
  const auto& felements_sum = fmatrix.GetElementsSum();
  const auto& selements_sum = smatrix.GetElementsSum();
  std::vector<std::thread> threads(ppc::util::GetPPCNumThreads());
  std::vector<MatrixComponents> threads_data(ppc::util::GetPPCNumThreads());
  auto function = [&](size_t start, size_t end, size_t index) {
    MatrixComponents thread_component;
    thread_component.m_elementsSum.resize(end - start);
    for (size_t i = start; i < end; ++i) {
      for (size_t j = 0; j < felements_sum.size(); ++j) {
        double sum =
            CalculateSum(fmatrix, smatrix, felements_sum, selements_sum, static_cast<int>(i), static_cast<int>(j));
        if (sum != 0) {
          thread_component.m_values.emplace_back(sum);
          thread_component.m_rows.emplace_back(j);
          thread_component.m_elementsSum[i - start]++;
        }
      }
    }
    threads_data[index] = std::move(thread_component);
  };
  auto indexes = CalculateSeparation(selements_sum.size());
  for (size_t i = 0; i < threads_data.size(); ++i) {
    threads[i] = std::thread(function, indexes[i].first, indexes[i].second, i);
  }
  std::ranges::for_each(threads, [&](auto& thread) { thread.join(); });
  MatrixComponents result;
  for (auto& data : threads_data) {
    for (size_t i = 0; i < data.m_rows.size(); ++i) {
      result.m_rows.emplace_back(data.m_rows[i]);
      result.m_values.emplace_back(data.m_values[i]);
    }
    for (size_t i = 0; i < data.m_elementsSum.size(); ++i) {
      result.m_elementsSum.emplace_back(data.m_elementsSum[i]);
    }
  }
  for (size_t i = 1; i < result.m_elementsSum.size(); ++i) {
    result.m_elementsSum[i] = result.m_elementsSum[i] + result.m_elementsSum[i - 1];
  }
  return {m_rowsCount_, smatrix.GetColumnsCount(), result};
}

SparseMatrix SparseMatrix::MatrixToSparse(int rows_count, int columns_count, const std::vector<double>& values) {
  MatrixComponents compontents;
  compontents.m_elementsSum.resize(columns_count);
  for (int i = 0; i < columns_count; ++i) {
    for (int j = 0; j < rows_count; ++j) {
      if (values[i + (columns_count * j)] != 0) {
        compontents.m_values.emplace_back(values[i + (columns_count * j)]);
        compontents.m_rows.emplace_back(j);
        compontents.m_elementsSum[i]++;
      }
    }
    if (i != columns_count - 1) {
      compontents.m_elementsSum[i + 1] = compontents.m_elementsSum[i];
    }
  }
  return {rows_count, columns_count, compontents};
}

std::vector<double> FromSparseMatrix(const SparseMatrix& matrix) {
  std::vector<double> simple_matrix(matrix.GetRowsCount() * matrix.GetColumnsCount());
  int counter = 0;
  for (size_t i = 0; i < matrix.GetElementsSum().size(); ++i) {
    auto limit = i == 0 ? matrix.GetElementsSum()[0] : matrix.GetElementsSum()[i] - matrix.GetElementsSum()[i - 1];
    for (int j = 0; j < limit; ++j) {
      simple_matrix[i + (matrix.GetColumnsCount() * matrix.GetRows()[counter])] = matrix.GetValues()[counter];
      counter++;
    }
  }
  return simple_matrix;
}

int SparseMatrix::GetElementsCount(int index, const std::vector<int>& elements_sum) {
  if (index == 0) {
    return elements_sum[index];
  }
  return elements_sum[index] - elements_sum[index - 1];
}

std::vector<double> BaseMatrixMultiplication(const std::vector<double>& fmatrix, int fmatrix_rows_count,
                                             int fmatrix_columns_count, const std::vector<double>& smatrix,
                                             int smatrix_rows_count, int smatrix_columns_count) {
  if (fmatrix_rows_count == smatrix_columns_count) {
    std::vector<double> answer(fmatrix_rows_count * smatrix_columns_count);
    for (int i = 0; i < fmatrix_rows_count; i++) {
      for (int j = 0; j < smatrix_columns_count; j++) {
        for (int n = 0; n < smatrix_rows_count; n++) {
          answer[j + (i * smatrix_columns_count)] +=
              fmatrix[(i * fmatrix_columns_count) + n] * smatrix[(n * smatrix_columns_count) + j];
        }
      }
    }
    return answer;
  }
  return {};
}
}  // namespace sadikov_i_sparse_matrix_multiplication_task_stl