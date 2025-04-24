#include "omp/muradov_m_rect_int/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

bool muradov_m_rect_int_omp::RectIntTaskOmp::ValidationImpl() {
  return task_data->inputs_count[0] == 1 && task_data->inputs_count[1] > 0 && task_data->outputs_count[0] == 1;
}

bool muradov_m_rect_int_omp::RectIntTaskOmp::PreProcessingImpl() {
  auto* p_grains = reinterpret_cast<int*>(task_data->inputs[0]);
  grains_ = *p_grains;

  auto* p_bounds = reinterpret_cast<std::pair<double, double>*>(task_data->inputs[1]);
  bounds_.assign(p_bounds, p_bounds + task_data->inputs_count[1]);

  return true;
}

bool muradov_m_rect_int_omp::RectIntTaskOmp::RunImpl() {
  res_ = 0;

  const auto dims = static_cast<std::size_t>(bounds_.size());

  double hh = 1.0;
  for (std::size_t i = 0; i < dims; i++) {
    hh *= (bounds_[i].second - bounds_[i].first) / grains_;
  }
  int pts = static_cast<int>(std::pow(grains_, dims));

  FunArgs args(dims);
  decltype(res_) lres{};
#pragma omp parallel for reduction(+ : lres) firstprivate(args)
  for (int i = 0; i < pts; i++) {
    auto j = i;
    for (size_t k = 0; k < dims; k++) {
      args[k] = bounds_[k].first;
      args[k] += (j % grains_) * (bounds_[k].second - bounds_[k].first) / grains_;
      j /= grains_;
    }
    lres += fun_(args);
  }

  res_ = lres * hh;

  return true;
}

bool muradov_m_rect_int_omp::RectIntTaskOmp::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = res_;
  return true;
}
