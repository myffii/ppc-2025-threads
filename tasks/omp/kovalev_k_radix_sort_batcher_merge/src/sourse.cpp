#include <omp.h>

#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "omp/kovalev_k_radix_sort_batcher_merge/include/header.hpp"

bool kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP::RadixUnsigned(unsigned long long* inp_arr,
                                                                           unsigned long long* temp,
                                                                           unsigned int size) {
  auto* masc = reinterpret_cast<unsigned char*>(inp_arr);
  int count[256];
  unsigned int sizetype = sizeof(unsigned long long);
  for (unsigned int i = 0; i < sizetype; i++) {
    Countbyte(inp_arr, count, i, size);
    for (unsigned int j = 0; j < size; j++) {
      temp[count[masc[(j * sizetype) + i]]++] = inp_arr[j];
    }
    memcpy(inp_arr, temp, sizeof(unsigned long long) * size);
  }
  return true;
}

bool kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP::RadixSigned(unsigned int start, unsigned int size) const {
  auto* mas = reinterpret_cast<unsigned long long*>(mas_ + start);
  auto* tmp = reinterpret_cast<unsigned long long*>(tmp_ + start);
  unsigned int count = 0;
  bool ret = RadixUnsigned(mas, tmp, size);
  while (count < size && mas_[start + count] >= 0) {
    count++;
  }
  if (count == size) {
    return ret;
  }
  memcpy(tmp, mas + count, sizeof(long long int) * (size - count));
  memcpy(tmp + (size - count), mas, sizeof(long long int) * (count));
  memcpy(mas, tmp, sizeof(long long int) * size);
  return ret;
}

bool kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP::Countbyte(unsigned long long* inp_arr, int* count,
                                                                       unsigned int byte, unsigned int size) {
  auto* masc = reinterpret_cast<unsigned char*>(inp_arr);
  unsigned int bias = sizeof(unsigned long long);
  for (unsigned int i = 0; i < 256; i++) {
    count[i] = 0;
  }
  for (unsigned int i = 0; i < size; i++) {
    count[masc[(i * bias) + byte]]++;
  }
  int tmp1 = count[0];
  count[0] = 0;
  for (unsigned int i = 1; i < 256; i++) {
    int tmp2 = count[i];
    count[i] = count[i - 1] + tmp1;
    tmp1 = tmp2;
  }
  return true;
}

bool kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP::OddEvenMerge(long long int* tmp, long long int* l,
                                                                          const long long int* r, unsigned int len_l,
                                                                          unsigned int len_r) {
  unsigned int iter_l = 0;
  unsigned int iter_r = 0;
  unsigned int iter_tmp = 0;

  while (iter_l < len_l && iter_r < len_r) {
    if (l[iter_l] < r[iter_r]) {
      tmp[iter_tmp] = l[iter_l];
      iter_l += 2;
    } else {
      tmp[iter_tmp] = r[iter_r];
      iter_r += 2;
    }

    iter_tmp += 2;
  }

  while (iter_l < len_l) {
    tmp[iter_tmp] = l[iter_l];
    iter_l += 2;
    iter_tmp += 2;
  }

  while (iter_r < len_r) {
    tmp[iter_tmp] = r[iter_r];
    iter_r += 2;
    iter_tmp += 2;
  }

  for (unsigned int i = 0; i < iter_tmp; i += 2) {
    l[i] = tmp[i];
  }
  return true;
}

bool kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP::FinalMerge() {
  unsigned int iter_even = 0;
  unsigned int iter_odd = 1;
  unsigned int iter_tmp = 0;

  while (iter_even < n_ && iter_odd < n_) {
    if (mas_[iter_even] < mas_[iter_odd]) {
      tmp_[iter_tmp] = mas_[iter_even];
      iter_even += 2;
    } else {
      tmp_[iter_tmp] = mas_[iter_odd];
      iter_odd += 2;
    }

    iter_tmp++;
  }

  while (iter_even < n_) {
    tmp_[iter_tmp] = mas_[iter_even];
    iter_even += 2;
    iter_tmp++;
  }

  while (iter_odd < n_) {
    tmp_[iter_tmp] = mas_[iter_odd];
    iter_odd += 2;
    iter_tmp++;
  }
  return true;
}

bool kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP::ValidationImpl() {
  return (task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == task_data->inputs_count[0]);
}

bool kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP::PreProcessingImpl() {
  n_input_ = task_data->inputs_count[0];

  effective_num_threads_ = static_cast<int>(std::pow(2, std::floor(std::log2(omp_get_max_threads()))));
  auto e_n_f = static_cast<unsigned int>(effective_num_threads_);
  n_ = n_input_ + (((2 * e_n_f) - n_input_ % (2 * e_n_f))) % (2 * e_n_f);
  loc_lenght_ = n_ / effective_num_threads_;

  mas_ = new long long int[n_];
  tmp_ = new long long int[n_];

  void* ptr_input = task_data->inputs[0];
  void* ptr_vec = mas_;
  memcpy(ptr_vec, ptr_input, sizeof(long long int) * n_input_);

  for (unsigned int i = n_input_; i < n_; i++) {
    mas_[i] = LLONG_MAX;
  }
  return true;
}

bool kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP::RunImpl() {
  if (static_cast<unsigned int>(omp_get_max_threads()) > 2 * n_input_) {
    bool ret = RadixSigned(0, n_input_);
    memcpy(tmp_, mas_, sizeof(long long int) * n_input_);
    return ret;
  }

  bool ret1 = true;
  bool ret2 = true;
#pragma omp parallel num_threads(effective_num_threads_)
  { ret1 = ret1 && RadixSigned(omp_get_thread_num() * loc_lenght_, loc_lenght_); }

  for (unsigned int i = effective_num_threads_; i > 1; i /= 2) {
#pragma omp parallel num_threads(i)
    {
      auto stride = static_cast<unsigned int>(omp_get_thread_num() / 2);
      unsigned int bias = omp_get_thread_num() % 2;
      unsigned int len = loc_lenght_ * (effective_num_threads_ / i);

      ret2 = ret2 && OddEvenMerge(tmp_ + (stride * 2 * len) + bias, mas_ + (stride * 2 * len) + bias,
                                  mas_ + (stride * 2 * len) + len + bias, len - bias, len - bias);
    }
  }
  FinalMerge();
  return (ret1 && ret2);
}

bool kovalev_k_radix_sort_batcher_merge_omp::TestTaskOpenMP::PostProcessingImpl() {
  memcpy(reinterpret_cast<long long int*>(task_data->outputs[0]), tmp_, sizeof(long long int) * n_input_);
  delete[] mas_;
  delete[] tmp_;
  return true;
}