#include "tbb/tyurin_m_matmul_crs_complex/include/ops_tbb.hpp"

#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/parallel_for.h"

namespace {
MatrixCRS TransposeMatrixCRS(const MatrixCRS &crs) {
  const auto new_cols = crs.GetRows();

  MatrixCRS res;
  res.cols_count = new_cols;
  res.rowptr.resize(crs.GetCols() + 2);
  res.colind.resize(crs.colind.size(), 0);
  res.data.resize(crs.data.size(), 0);

  for (uint32_t i = 0; i < crs.data.size(); ++i) {
    ++res.rowptr[crs.colind[i] + 2];
  }
  for (uint32_t i = 2; i < res.rowptr.size(); ++i) {
    res.rowptr[i] += res.rowptr[i - 1];
  }
  for (uint32_t i = 0; i < new_cols; ++i) {
    for (uint32_t j = crs.rowptr[i]; j < crs.rowptr[i + 1]; ++j) {
      const auto new_index = res.rowptr[crs.colind[j] + 1]++;
      res.data[new_index] = crs.data[j];
      res.colind[new_index] = i;
    }
  }
  res.rowptr.pop_back();

  return res;
}
}  // namespace

bool tyurin_m_matmul_crs_complex_tbb::TestTaskTbb::ValidationImpl() {
  const bool left_cols_equal_right_rows = task_data->inputs_count[1] == task_data->inputs_count[2];
  const bool there_are_rows_and_cols =
      task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 && task_data->inputs_count[2] > 0;
  return left_cols_equal_right_rows && there_are_rows_and_cols && task_data->outputs_count[0] == 1;
}

bool tyurin_m_matmul_crs_complex_tbb::TestTaskTbb::PreProcessingImpl() {
  lhs_ = *reinterpret_cast<MatrixCRS *>(task_data->inputs[0]);
  rhs_ = TransposeMatrixCRS(*reinterpret_cast<MatrixCRS *>(task_data->inputs[1]));
  res_ = {};
  res_.rowptr.resize(lhs_.GetRows() + 1);
  res_.cols_count = rhs_.GetRows();
  return true;
}

bool tyurin_m_matmul_crs_complex_tbb::TestTaskTbb::RunImpl() {  // NOLINT(readability-function-cognitive-complexity)
  const auto rows = lhs_.GetRows();
  const auto cols = rhs_.GetRows();

  std::vector<std::vector<std::tuple<std::complex<double>, uint32_t>>> buf(rows);

  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  arena.execute([&] {
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, rows),
                              [&](const tbb::blocked_range<std::size_t> &r) {
                                for (uint32_t i = r.begin(); i < r.end(); ++i) {
                                  for (uint32_t j = 0; j < cols; ++j) {
                                    auto ii = lhs_.rowptr[i];
                                    auto ij = rhs_.rowptr[j];
                                    std::complex<double> summul = 0.0;
                                    while (ii < lhs_.rowptr[i + 1] && ij < rhs_.rowptr[j + 1]) {
                                      if (lhs_.colind[ii] < rhs_.colind[ij]) {
                                        ++ii;
                                      } else if (lhs_.colind[ii] > rhs_.colind[ij]) {
                                        ++ij;
                                      } else {
                                        summul += lhs_.data[ii++] * rhs_.data[ij++];
                                      }
                                    }
                                    if (summul != 0.0) {
                                      buf[i].emplace_back(summul, j);
                                    }
                                  }
                                }
                              });
    return;
  });

  for (uint32_t i = 0; i < rows; i++) {
    res_.rowptr[i + 1] = res_.rowptr[i];
    for (const auto &[summul, j] : buf[i]) {
      res_.data.push_back(summul);
      res_.colind.push_back(j);
      ++res_.rowptr[i + 1];
    }
  }

  return true;
}

bool tyurin_m_matmul_crs_complex_tbb::TestTaskTbb::PostProcessingImpl() {
  *reinterpret_cast<MatrixCRS *>(task_data->outputs[0]) = res_;
  return true;
}
