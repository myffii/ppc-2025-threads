#include "stl/filateva_e_simpson/include/ops_stl.hpp"

#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool filateva_e_simpson_stl::Simpson::PreProcessingImpl() {
  mer_ = task_data->inputs_count[0];
  steps_ = task_data->inputs_count[1];

  auto *temp_a = reinterpret_cast<double *>(task_data->inputs[0]);
  a_.insert(a_.end(), temp_a, temp_a + mer_);

  auto *temp_b = reinterpret_cast<double *>(task_data->inputs[1]);
  b_.insert(b_.end(), temp_b, temp_b + mer_);

  f_ = reinterpret_cast<Func>(task_data->inputs[2]);

  return true;
}

bool filateva_e_simpson_stl::Simpson::ValidationImpl() {
  size_t mer = task_data->inputs_count[0];
  auto *temp_a = reinterpret_cast<double *>(task_data->inputs[0]);
  auto *temp_b = reinterpret_cast<double *>(task_data->inputs[1]);
  if (task_data->inputs_count[1] % 2 == 1) {
    return false;
  }
  for (size_t i = 0; i < mer; i++) {
    if (temp_b[i] <= temp_a[i]) {
      return false;
    }
  }
  return true;
}

double filateva_e_simpson_stl::Simpson::IntegralFunc(unsigned long start, unsigned long end) {
  double local_res = 0.0;
  for (unsigned long i = start; i < end; i++) {
    unsigned long temp = i;
    std::vector<double> param(mer_);
    double weight = 1.0;

    for (size_t m = 0; m < mer_; m++) {
      size_t shag_i = temp % (steps_ + 1);
      temp /= (steps_ + 1);

      param[m] = a_[m] + h_[m] * static_cast<double>(shag_i);

      if (shag_i == 0 || shag_i == steps_) {
        continue;
      }
      weight *= (2.0 + static_cast<double>(shag_i % 2) * 2);
    }

    local_res += weight * f_(param);
  }
  return local_res;
}

bool filateva_e_simpson_stl::Simpson::RunImpl() {
  h_.resize(mer_);
  for (size_t i = 0; i < mer_; i++) {
    h_[i] = static_cast<double>(b_[i] - a_[i]) / static_cast<double>(steps_);
  }

  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads(num_threads);
  std::vector<double> local_res(num_threads, 0.0);
  res_ = 0.0;

  unsigned long del = (unsigned long)std::pow(steps_ + 1, mer_) / num_threads;
  unsigned long ost = (unsigned long)std::pow(steps_ + 1, mer_) % num_threads;
  for (int i = 0; i < num_threads; i++) {
    threads[i] = std::thread([&, i]() { local_res[i] = IntegralFunc(ost + (del * i), ost + (del * (i + 1))); });
  }
  res_ += IntegralFunc(0, ost);
  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
    res_ += local_res[i];
  }

  for (size_t i = 0; i < mer_; i++) {
    res_ *= (h_[i] / 3.0);
  }
  return true;
}

bool filateva_e_simpson_stl::Simpson::PostProcessingImpl() {
  reinterpret_cast<double *>(task_data->outputs[0])[0] = res_;
  return true;
}
