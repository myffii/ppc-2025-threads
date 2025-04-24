#include "tbb/karaseva_e_congrad/include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

namespace karaseva_e_congrad_tbb {

bool TestTaskTBB::PreProcessingImpl() {
  // Initialize problem size from input data
  size_ = task_data->inputs_count[1];

  // Map raw input pointers to matrices/vectors
  auto* a_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* b_ptr = reinterpret_cast<double*>(task_data->inputs[1]);

  // Create contiguous storage for matrix A and vector b
  A_ = std::vector<double>(a_ptr, a_ptr + (size_ * size_));
  b_ = std::vector<double>(b_ptr, b_ptr + size_);

  // Initial guess x0 = 0
  x_ = std::vector<double>(size_, 0.0);

  return true;
}

bool TestTaskTBB::ValidationImpl() {
  const bool valid_input = task_data->inputs_count[0] == task_data->inputs_count[1] * task_data->inputs_count[1];
  const bool valid_output = task_data->outputs_count[0] == task_data->inputs_count[1];
  return valid_input && valid_output;
}

namespace {

// Helper function to compute dot product of two vectors using TBB
double ComputeDotProduct(const std::vector<double>& vec1, const std::vector<double>& vec2, size_t size) {
  return tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, size), 0.0,
      [&](const tbb::blocked_range<size_t>& range, double local_sum) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          local_sum += vec1[i] * vec2[i];
        }
        return local_sum;
      },
      std::plus<>());
}

// Helper function for matrix-vector multiplication using TBB
void MatrixVectorMultiply(const std::vector<double>& matrix, const std::vector<double>& vec,
                          std::vector<double>& result, size_t size) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      double temp = 0.0;
      for (size_t j = 0; j < size; ++j) {
        temp += matrix[(i * size) + j] * vec[j];
      }
      result[i] = temp;
    }
  });
}

// Helper function to update solution and residual vectors
void UpdateVectors(std::vector<double>& x, std::vector<double>& r, double alpha, const std::vector<double>& p,
                   const std::vector<double>& ap, const tbb::blocked_range<size_t>& range) {
  for (size_t i = range.begin(); i != range.end(); ++i) {
    x[i] += alpha * p[i];
    r[i] -= alpha * ap[i];
  }
}

// Helper function to update search direction vector
void UpdateSearchDirection(std::vector<double>& p, const std::vector<double>& r, double beta,
                           const tbb::blocked_range<size_t>& range) {
  for (size_t i = range.begin(); i != range.end(); ++i) {
    p[i] = r[i] + beta * p[i];
  }
}

}  // namespace

bool TestTaskTBB::RunImpl() {
  std::vector<double> r(size_);
  std::vector<double> p(size_);
  std::vector<double> ap(size_);

  // Parallel initialization of r and p vectors
  tbb::parallel_for(tbb::blocked_range<size_t>(0, size_), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      r[i] = b_[i];
      p[i] = r[i];
    }
  });

  double rs_old = ComputeDotProduct(r, r, size_);
  const double tolerance = 1e-10;
  const size_t max_iterations = size_;

  for (size_t k = 0; k < max_iterations; ++k) {
    MatrixVectorMultiply(A_, p, ap, size_);

    const double p_ap = ComputeDotProduct(p, ap, size_);
    if (std::fabs(p_ap) < 1e-15) {
      break;
    }

    const double alpha = rs_old / p_ap;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, size_),
                      [&](const tbb::blocked_range<size_t>& range) { UpdateVectors(x_, r, alpha, p, ap, range); });

    const double rs_new = ComputeDotProduct(r, r, size_);
    if (rs_new < tolerance * tolerance) {
      break;
    }

    const double beta = rs_new / rs_old;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, size_),
                      [&](const tbb::blocked_range<size_t>& range) { UpdateSearchDirection(p, r, beta, range); });

    rs_old = rs_new;
  }

  return true;
}

bool TestTaskTBB::PostProcessingImpl() {
  auto* x_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, x_.size()), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      x_ptr[i] = x_[i];
    }
  });
  return true;
}

}  // namespace karaseva_e_congrad_tbb