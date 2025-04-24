#include "omp/korotin_e_crs_multiplication/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool korotin_e_crs_multiplication_omp::CrsMultiplicationOMP::PreProcessingImpl() {
  A_N_ = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[0]);
  A_rI_ = std::vector<unsigned int>(in_ptr, in_ptr + A_N_);

  A_Nz_ = task_data->inputs_count[1];
  in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[1]);
  A_col_ = std::vector<unsigned int>(in_ptr, in_ptr + A_Nz_);

  auto *val_ptr = reinterpret_cast<double *>(task_data->inputs[2]);
  A_val_ = std::vector<double>(val_ptr, val_ptr + A_Nz_);

  B_N_ = task_data->inputs_count[3];
  in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[3]);
  B_rI_ = std::vector<unsigned int>(in_ptr, in_ptr + B_N_);

  B_Nz_ = task_data->inputs_count[4];
  in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[4]);
  B_col_ = std::vector<unsigned int>(in_ptr, in_ptr + B_Nz_);

  val_ptr = reinterpret_cast<double *>(task_data->inputs[5]);
  B_val_ = std::vector<double>(val_ptr, val_ptr + B_Nz_);

  unsigned int output_size = task_data->outputs_count[0];
  output_rI_ = std::vector<unsigned int>(output_size);

  return true;
}

bool korotin_e_crs_multiplication_omp::CrsMultiplicationOMP::ValidationImpl() {
  return task_data->inputs_count[1] == task_data->inputs_count[2] &&
         task_data->inputs_count[4] == task_data->inputs_count[5] &&
         task_data->inputs_count[0] == task_data->outputs_count[0] &&
         *std::max_element(reinterpret_cast<unsigned int *>(task_data->inputs[1]),
                           reinterpret_cast<unsigned int *>(task_data->inputs[1]) + task_data->inputs_count[1]) <=
             task_data->inputs_count[3] - 2;
}

bool korotin_e_crs_multiplication_omp::CrsMultiplicationOMP::RunImpl() {
  std::vector<unsigned int> tr_i(*std::ranges::max_element(B_col_.begin(), B_col_.end()) + 2, 0);
  unsigned int i = 0;
  unsigned int j = 0;
  for (i = 0; i < B_Nz_; i++) {
    tr_i[B_col_[i] + 1]++;
  }
  for (i = 1; i < tr_i.size(); i++) {
    tr_i[i] += tr_i[i - 1];
  }

  std::vector<unsigned int> tcol(B_Nz_, 0);
  std::vector<double> tval(B_Nz_, 0);
  for (i = 0; i < B_N_ - 1; i++) {
    for (j = B_rI_[i]; j < B_rI_[i + 1]; j++) {
      tval[tr_i[B_col_[j]]] = B_val_[j];
      tcol[tr_i[B_col_[j]]] = i;
      tr_i[B_col_[j]]++;
    }
  }
  for (i = tr_i.size() - 1; i > 0; i--) {
    tr_i[i] = tr_i[i - 1];
  }
  tr_i[0] = 0;

  unsigned int ai = 0;
  unsigned int bt = 0;
  double sum = 0;
  std::ranges::fill(output_rI_.begin(), output_rI_.end(), 0);
  output_col_.clear();
  output_val_.clear();
  std::vector<std::vector<double>> local_val(omp_get_max_threads());
  std::vector<std::vector<unsigned int>> local_col(omp_get_max_threads());
  std::vector<int> temp_r_i(A_N_, 0);

#pragma omp parallel for private(j, sum, ai, bt)
  for (int k = 0; k < static_cast<int>(A_N_ - 1); k++) {
    int thread_id = omp_get_thread_num();
    for (j = 0; j < tr_i.size() - 1; j++) {
      sum = 0;
      ai = A_rI_[k];
      bt = tr_i[j];
      while (ai < A_rI_[k + 1] && bt < tr_i[j + 1]) {
        if (A_col_[ai] == tcol[bt]) {
          sum += A_val_[ai] * tval[bt];
          ai++;
          bt++;
        } else if (A_col_[ai] < tcol[bt]) {
          ai++;
        } else {
          bt++;
        }
      }
      if (sum != 0) {
        local_val[thread_id].push_back(sum);
        local_col[thread_id].push_back(j);
        temp_r_i[k + 1]++;
      }
    }
  }

  for (int t = 0; t < omp_get_max_threads(); t++) {
    output_val_.insert(output_val_.end(), local_val[t].begin(), local_val[t].end());
    output_col_.insert(output_col_.end(), local_col[t].begin(), local_col[t].end());
  }

  for (i = 1; i < A_N_; i++) {
    output_rI_[i] += output_rI_[i - 1] + temp_r_i[i];
  }
  return true;
}

bool korotin_e_crs_multiplication_omp::CrsMultiplicationOMP::PostProcessingImpl() {
  for (size_t i = 0; i < output_rI_.size(); i++) {
    reinterpret_cast<unsigned int *>(task_data->outputs[0])[i] = output_rI_[i];
  }
  for (size_t i = 0; i < output_col_.size(); i++) {
    reinterpret_cast<unsigned int *>(task_data->outputs[1])[i] = output_col_[i];
    reinterpret_cast<double *>(task_data->outputs[2])[i] = output_val_[i];
  }
  task_data->outputs_count.emplace_back(output_col_.size());
  task_data->outputs_count.emplace_back(output_val_.size());
  return true;
}
