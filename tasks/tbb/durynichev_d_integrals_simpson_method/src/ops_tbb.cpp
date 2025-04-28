#include "tbb/durynichev_d_integrals_simpson_method/include/ops_tbb.hpp"

#include <cmath>
#include <vector>

#include "oneapi/tbb/mutex.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

namespace durynichev_d_integrals_simpson_method_tbb {

bool SimpsonIntegralTBB::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  boundaries_ = std::vector<double>(in_ptr, in_ptr + input_size);
  n_ = static_cast<int>(boundaries_.back());
  boundaries_.pop_back();
  dim_ = boundaries_.size() / 2;

  result_ = 0.0;
  return true;
}

bool SimpsonIntegralTBB::ValidationImpl() {
  return task_data->inputs_count[0] >= 3 && task_data->outputs_count[0] == 1 && (n_ % 2 == 0);
}

bool SimpsonIntegralTBB::RunImpl() {
  oneapi::tbb::task_arena arena(oneapi::tbb::task_arena::automatic);
  arena.execute([&] {
    tbb::task_group tg;
    tg.run([&] {
      if (dim_ == 1) {
        result_ = Simpson1D(boundaries_[0], boundaries_[1]);
      } else if (dim_ == 2) {
        result_ = Simpson2D(boundaries_[0], boundaries_[1], boundaries_[2], boundaries_[3]);
      }
    });
    tg.wait();
  });
  return true;
}

bool SimpsonIntegralTBB::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}

double SimpsonIntegralTBB::Func1D(double x) { return x * x; }

double SimpsonIntegralTBB::Func2D(double x, double y) { return (x * x) + (y * y); }

double SimpsonIntegralTBB::Simpson1D(double a, double b) const {
  double h = (b - a) / n_;
  double sum = Func1D(a) + Func1D(b);

  double inner_sum = 0.0;
  tbb::parallel_for(tbb::blocked_range<int>(1, n_), [&](const tbb::blocked_range<int>& r) {
    double local_sum = 0.0;
    for (int i = r.begin(); i < r.end(); ++i) {
      double coef = (i % 2 == 0) ? 2.0 : 4.0;
      local_sum += coef * Func1D(a + (i * h));
    }
    tbb::mutex::scoped_lock lock(mutex_);
    inner_sum += local_sum;
  });

  sum += inner_sum;
  return sum * h / 3.0;
}

double SimpsonIntegralTBB::Simpson2D(double x0, double x1, double y0, double y1) const {
  double hx = (x1 - x0) / n_;
  double hy = (y1 - y0) / n_;
  double sum = 0.0;

  tbb::parallel_for(tbb::blocked_range<int>(0, n_ + 1), [&](const tbb::blocked_range<int>& r) {
    double local_sum = 0.0;
    for (int i = r.begin(); i < r.end(); ++i) {
      double x = x0 + (i * hx);
      double coef_x = 0;
      if (i == 0 || i == n_) {
        coef_x = 1.0;
      } else if (i % 2 != 0) {
        coef_x = 4.0;
      } else {
        coef_x = 2.0;
      }

      for (int j = 0; j <= n_; ++j) {
        double y = y0 + (j * hy);
        double coef_y = 0;
        if (j == 0 || j == n_) {
          coef_y = 1.0;
        } else if (j % 2 != 0) {
          coef_y = 4.0;
        } else {
          coef_y = 2.0;
        }
        local_sum += coef_x * coef_y * Func2D(x, y);
      }
    }
    tbb::mutex::scoped_lock lock(mutex_);
    sum += local_sum;
  });

  return sum * hx * hy / 9.0;
}

}  // namespace durynichev_d_integrals_simpson_method_tbb