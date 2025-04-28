#include "tbb/solovev_a_ccs_mmult_sparse/include/ccs_mmult_sparse_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <complex>
#include <vector>

void solovev_a_matrix_tbb::TBBMatMultCcs::ComputeColumnSizes() {
  tbb::parallel_for(tbb::blocked_range<int>(0, M3_->c_n), [&](const tbb::blocked_range<int>& r) {
    for (int m2_c = r.begin(); m2_c != r.end(); ++m2_c) {
      std::vector<int> available_el(M3_->r_n, 0);
      for (int m2_i = M2_->col_p[m2_c]; m2_i < M2_->col_p[m2_c + 1]; ++m2_i) {
        int m2_r = M2_->row[m2_i];
        for (int m1_i = M1_->col_p[m2_r]; m1_i < M1_->col_p[m2_r + 1]; ++m1_i) {
          available_el[M1_->row[m1_i]] = 1;
        }
      }

      int n_z_c_cnt = 0;
      for (int m3_r = 0; m3_r < M3_->r_n; ++m3_r) {
        n_z_c_cnt += available_el[m3_r];
      }

      M3_->col_p[m2_c + 1] = n_z_c_cnt;
    }
  });
}

void solovev_a_matrix_tbb::TBBMatMultCcs::FillMatrixValues() {
  tbb::parallel_for(tbb::blocked_range<int>(0, M3_->c_n), [&](const tbb::blocked_range<int>& r) {
    for (int m2_c = r.begin(); m2_c != r.end(); ++m2_c) {
      std::vector<int> available_el(M3_->r_n, 0);
      std::vector<std::complex<double>> cask(M3_->r_n, std::complex<double>(0.0, 0.0));

      for (int m2_i = M2_->col_p[m2_c]; m2_i < M2_->col_p[m2_c + 1]; ++m2_i) {
        int m2_r = M2_->row[m2_i];
        std::complex<double> m2_val = M2_->val[m2_i];

        for (int m1_i = M1_->col_p[m2_r]; m1_i < M1_->col_p[m2_r + 1]; ++m1_i) {
          int m1_row = M1_->row[m1_i];
          cask[m1_row] += M1_->val[m1_i] * m2_val;
          available_el[m1_row] = 1;
        }
      }

      int c_pos = M3_->col_p[m2_c];
      for (int m3_r = 0; m3_r < M3_->r_n; ++m3_r) {
        if (available_el[m3_r]) {
          M3_->row[c_pos] = m3_r;
          M3_->val[c_pos++] = cask[m3_r];
        }
      }
    }
  });
}

bool solovev_a_matrix_tbb::TBBMatMultCcs::PreProcessingImpl() {
  M1_ = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[0]);
  M2_ = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[1]);
  M3_ = reinterpret_cast<MatrixInCcsSparse*>(task_data->outputs[0]);
  return true;
}

bool solovev_a_matrix_tbb::TBBMatMultCcs::ValidationImpl() {
  auto* m1_c_n = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[0]);
  auto* m2_r_n = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[1]);
  return (m1_c_n->c_n == m2_r_n->r_n);
}

bool solovev_a_matrix_tbb::TBBMatMultCcs::RunImpl() {
  M3_->r_n = M1_->r_n;
  M3_->c_n = M2_->c_n;
  M3_->col_p.resize(M3_->c_n + 1);
  M3_->col_p[0] = 0;

  ComputeColumnSizes();

  for (int i = 1; i <= M3_->c_n; ++i) {
    M3_->col_p[i] += M3_->col_p[i - 1];
  }

  int n_z_full = M3_->col_p[M3_->c_n];
  M3_->n_z = n_z_full;
  M3_->row.resize(n_z_full);
  M3_->val.resize(n_z_full);

  FillMatrixValues();

  return true;
}

bool solovev_a_matrix_tbb::TBBMatMultCcs::PostProcessingImpl() { return true; }
