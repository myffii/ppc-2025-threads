#include "stl/lavrentiev_A_CCS/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

lavrentiev_a_ccs_stl::Sparse lavrentiev_a_ccs_stl::CCSSTL::ConvertToSparse(std::pair<int, int> size,
                                                                           const std::vector<double> &values) {
  auto [nsize, elements_and_rows, columns_sum] = Sparse();
  columns_sum.resize(size.second);
  for (int i = 0; i < size.second; ++i) {
    for (int j = 0; j < size.first; ++j) {
      if (values[i + (size.second * j)] != 0) {
        elements_and_rows.emplace_back(values[i + (size.second * j)], j);
        columns_sum[i] += 1;
      }
    }
    if (i != size.second - 1) {
      columns_sum[i + 1] = columns_sum[i];
    }
  }
  return {.size = size, .elements_and_rows = elements_and_rows, .columnsSum = columns_sum};
}

lavrentiev_a_ccs_stl::Sparse lavrentiev_a_ccs_stl::CCSSTL::Transpose(const Sparse &sparse) {
  auto [size, elements_and_rows, columns_sum] = Sparse();
  size.first = sparse.size.second;
  size.second = sparse.size.first;
  int need_size = std::max(sparse.size.first, sparse.size.second);
  std::vector<std::vector<std::pair<double, int>>> new_elements_and_rows(need_size);
  int counter = 0;
  for (int i = 0; i < static_cast<int>(sparse.columnsSum.size()); ++i) {
    for (int j = 0; j < GetElementsCount(i, sparse.columnsSum); ++j) {
      new_elements_and_rows[sparse.elements_and_rows[counter].second].emplace_back(
          sparse.elements_and_rows[counter].first, i);
      counter++;
    }
  }
  elements_and_rows.reserve(counter);
  for (int i = 0; i < static_cast<int>(new_elements_and_rows.size()); ++i) {
    for (int j = 0; j < static_cast<int>(new_elements_and_rows[i].size()); ++j) {
      elements_and_rows.emplace_back(new_elements_and_rows[i][j]);
    }
    i > 0 ? columns_sum.emplace_back(new_elements_and_rows[i].size() + columns_sum[i - 1])
          : columns_sum.emplace_back(new_elements_and_rows[i].size());
  }
  return {.size = size, .elements_and_rows = elements_and_rows, .columnsSum = columns_sum};
}

int lavrentiev_a_ccs_stl::CCSSTL::CalculateStartIndex(int index, const std::vector<int> &columns_sum) {
  return index == 0 ? 0 : columns_sum[index] - GetElementsCount(index, columns_sum);
}

lavrentiev_a_ccs_stl::Sparse lavrentiev_a_ccs_stl::CCSSTL::MatMul(const Sparse &matrix1, const Sparse &matrix2) {
  Sparse temporary_matrix;
  std::vector<std::thread> threads(ppc::util::GetPPCNumThreads());
  temporary_matrix.columnsSum.resize(matrix2.size.second);
  temporary_matrix.elements_and_rows.resize((matrix2.columnsSum.size() * matrix1.columnsSum.size()) +
                                            std::max(matrix1.columnsSum.size(), matrix2.columnsSum.size()));
  auto transposed_matrix = Transpose(matrix1);
  auto accumulate = [&](int i_index, int j_index) {
    double sum = 0.0;
    for (int x = 0; x < GetElementsCount(j_index, transposed_matrix.columnsSum); x++) {
      for (int y = 0; y < GetElementsCount(i_index, matrix2.columnsSum); y++) {
        if (transposed_matrix.elements_and_rows[CalculateStartIndex(j_index, transposed_matrix.columnsSum) + x]
                .second == matrix2.elements_and_rows[CalculateStartIndex(i_index, matrix2.columnsSum) + y].second) {
          sum += transposed_matrix.elements_and_rows[x + CalculateStartIndex(j_index, transposed_matrix.columnsSum)]
                     .first *
                 matrix2.elements_and_rows[y + CalculateStartIndex(i_index, matrix2.columnsSum)].first;
        }
      }
    }
    return sum;
  };
  auto matrix_multiplicator = [&](int begin, int end) {
    for (int i = begin; i != end; ++i) {
      for (int j = 0; j < static_cast<int>(transposed_matrix.columnsSum.size()); ++j) {
        double s = accumulate(i, j);
        if (s != 0) {
          temporary_matrix.elements_and_rows[(i * matrix2.size.second) + j] = {s, j};
          temporary_matrix.columnsSum[i]++;
        }
      }
    }
  };
  int thread_data_amount = static_cast<int>(matrix2.columnsSum.size()) / ppc::util::GetPPCNumThreads();
  for (size_t i = 0; i < threads.size(); ++i) {
    if (i != threads.size() - 1) {
      threads[i] = std::thread(matrix_multiplicator, i * thread_data_amount, (i + 1) * thread_data_amount);
    } else {
      threads[i] =
          std::thread(matrix_multiplicator, i * thread_data_amount,
                      ((i + 1) * thread_data_amount) + (matrix2.columnsSum.size() % ppc::util::GetPPCNumThreads()));
    }
  }
  std::ranges::for_each(threads, [&](std::thread &thread) { thread.join(); });
  for (size_t i = 1; i < temporary_matrix.columnsSum.size(); ++i) {
    temporary_matrix.columnsSum[i] = temporary_matrix.columnsSum[i] + temporary_matrix.columnsSum[i - 1];
  }
  temporary_matrix.size.first = matrix2.size.second;
  temporary_matrix.size.second = matrix2.size.second;
  std::erase_if(temporary_matrix.elements_and_rows, [](auto &current_element) { return current_element.first == 0.0; });
  return {.size = temporary_matrix.size,
          .elements_and_rows = temporary_matrix.elements_and_rows,
          .columnsSum = temporary_matrix.columnsSum};
}

int lavrentiev_a_ccs_stl::CCSSTL::GetElementsCount(int index, const std::vector<int> &columns_sum) {
  if (index == 0) {
    return columns_sum[index];
  }
  return columns_sum[index] - columns_sum[index - 1];
}

std::vector<double> lavrentiev_a_ccs_stl::CCSSTL::ConvertFromSparse(const Sparse &matrix) {
  std::vector<double> nmatrix(matrix.size.first * matrix.size.second);
  int counter = 0;
  for (size_t i = 0; i < matrix.columnsSum.size(); ++i) {
    for (int j = 0; j < GetElementsCount(static_cast<int>(i), matrix.columnsSum); ++j) {
      nmatrix[i + (matrix.size.second * matrix.elements_and_rows[counter].second)] =
          matrix.elements_and_rows[counter].first;
      counter++;
    }
  }
  return nmatrix;
}

bool lavrentiev_a_ccs_stl::CCSSTL::PreProcessingImpl() {
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

bool lavrentiev_a_ccs_stl::CCSSTL::IsEmpty() const {
  return A_.size.first * A_.size.second == 0 || B_.size.first * B_.size.second == 0;
}

bool lavrentiev_a_ccs_stl::CCSSTL::ValidationImpl() {
  return task_data->inputs_count[0] * task_data->inputs_count[3] == task_data->outputs_count[0] &&
         task_data->inputs_count[0] == task_data->inputs_count[3] &&
         task_data->inputs_count[1] == task_data->inputs_count[2];
}

bool lavrentiev_a_ccs_stl::CCSSTL::RunImpl() {
  Answer_ = MatMul(A_, B_);
  return true;
}

bool lavrentiev_a_ccs_stl::CCSSTL::PostProcessingImpl() {
  std::ranges::copy(ConvertFromSparse(Answer_), reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}