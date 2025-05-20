#include "all/nasedkin_e_strassen_algorithm/include/ops_all.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "core/util/include/util.hpp"

namespace nasedkin_e_strassen_algorithm_all {

bool StrassenAll::PreProcessingImpl() {
  if (world_.rank() == 0) {
    unsigned int input_size = task_data->inputs_count[0];
    auto *in_ptr_a = reinterpret_cast<double *>(task_data->inputs[0]);
    auto *in_ptr_b = reinterpret_cast<double *>(task_data->inputs[1]);

    matrix_size_ = static_cast<int>(std::sqrt(input_size));
    input_matrix_a_.resize(matrix_size_ * matrix_size_);
    input_matrix_b_.resize(matrix_size_ * matrix_size_);

    std::ranges::copy(in_ptr_a, in_ptr_a + input_size, input_matrix_a_.begin());
    std::ranges::copy(in_ptr_b, in_ptr_b + input_size, input_matrix_b_.begin());

    if ((matrix_size_ & (matrix_size_ - 1)) != 0) {
      original_size_ = matrix_size_;
      input_matrix_a_ = PadMatrixToPowerOfTwo(input_matrix_a_, matrix_size_);
      input_matrix_b_ = PadMatrixToPowerOfTwo(input_matrix_b_, matrix_size_);
      matrix_size_ = static_cast<int>(std::sqrt(input_matrix_a_.size()));
    } else {
      original_size_ = matrix_size_;
    }

    output_matrix_.resize(matrix_size_ * matrix_size_, 0.0);
  }
  return true;
}

bool StrassenAll::ValidationImpl() {
  bool valid = true;
  if (world_.rank() == 0) {
    unsigned int input_size_a = task_data->inputs_count[0];
    unsigned int input_size_b = task_data->inputs_count[1];
    unsigned int output_size = task_data->outputs_count[0];

    if (input_size_a == 0 || input_size_b == 0 || output_size == 0) {
      valid = false;
    } else {
      int size_a = static_cast<int>(std::sqrt(input_size_a));
      int size_b = static_cast<int>(std::sqrt(input_size_b));
      int size_output = static_cast<int>(std::sqrt(output_size));
      valid = (size_a == size_b) && (size_a == size_output);
    }
  }
  boost::mpi::broadcast(world_, valid, 0);
  return valid;
}

bool StrassenAll::RunImpl() {
  boost::mpi::broadcast(world_, matrix_size_, 0);
  boost::mpi::broadcast(world_, original_size_, 0);
  if (world_.rank() != 0) {
    input_matrix_a_.resize(matrix_size_ * matrix_size_);
    input_matrix_b_.resize(matrix_size_ * matrix_size_);
    output_matrix_.resize(matrix_size_ * matrix_size_, 0.0);
  }
  boost::mpi::broadcast(world_, input_matrix_a_, 0);
  boost::mpi::broadcast(world_, input_matrix_b_, 0);

  constexpr int kNumProds = 7;
  int half_size = matrix_size_ / 2;
  std::vector<double> a11(half_size * half_size);
  std::vector<double> a12(half_size * half_size);
  std::vector<double> a21(half_size * half_size);
  std::vector<double> a22(half_size * half_size);
  std::vector<double> b11(half_size * half_size);
  std::vector<double> b12(half_size * half_size);
  std::vector<double> b21(half_size * half_size);
  std::vector<double> b22(half_size * half_size);

  SplitMatrix(input_matrix_a_, a11, 0, 0, matrix_size_);
  SplitMatrix(input_matrix_a_, a12, 0, half_size, matrix_size_);
  SplitMatrix(input_matrix_a_, a21, half_size, 0, matrix_size_);
  SplitMatrix(input_matrix_a_, a22, half_size, half_size, matrix_size_);
  SplitMatrix(input_matrix_b_, b11, 0, 0, matrix_size_);
  SplitMatrix(input_matrix_b_, b12, 0, half_size, matrix_size_);
  SplitMatrix(input_matrix_b_, b21, half_size, 0, matrix_size_);
  SplitMatrix(input_matrix_b_, b22, half_size, half_size, matrix_size_);

  std::vector<std::vector<double>> products(kNumProds, std::vector<double>(half_size * half_size));
  std::mutex mtx;
  std::vector<std::thread> threads;

  for (size_t i = world_.rank(); i < kNumProds; i += world_.size()) {
    threads.emplace_back(&StrassenAll::StrassenWorker, i, std::cref(input_matrix_a_), std::cref(input_matrix_b_),
                         matrix_size_, std::ref(products[i]), std::ref(mtx));
  }
  for (auto &t : threads) {
    t.join();
  }

  if (world_.rank() == 0) {
    std::vector<std::vector<double>> all_products(kNumProds, std::vector<double>(half_size * half_size));
    for (size_t i = 0; i < kNumProds; ++i) {
      if (i % world_.size() == 0) {
        all_products[i] = products[i];
      } else {
        size_t src = i % world_.size();
        if (i > static_cast<size_t>(std::numeric_limits<int>::max())) {
          throw std::runtime_error("Tag value too large for int");
        }
        world_.recv(static_cast<int>(src), static_cast<int>(i), all_products[i]);
      }
    }

    std::vector<double> c11 = AddMatrices(
        SubtractMatrices(AddMatrices(all_products[0], all_products[3], half_size), all_products[4], half_size),
        all_products[6], half_size);
    std::vector<double> c12 = AddMatrices(all_products[2], all_products[4], half_size);
    std::vector<double> c21 = AddMatrices(all_products[1], all_products[3], half_size);
    std::vector<double> c22 = AddMatrices(
        SubtractMatrices(AddMatrices(all_products[0], all_products[2], half_size), all_products[1], half_size),
        all_products[5], half_size);

    output_matrix_.resize(matrix_size_ * matrix_size_);
    MergeMatrix(output_matrix_, c11, 0, 0, matrix_size_);
    MergeMatrix(output_matrix_, c12, 0, half_size, matrix_size_);
    MergeMatrix(output_matrix_, c21, half_size, 0, matrix_size_);
    MergeMatrix(output_matrix_, c22, half_size, half_size, matrix_size_);
  } else {
    for (size_t i = world_.rank(); i < kNumProds; i += world_.size()) {
      if (i > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Tag value too large for int");
      }
      world_.send(0, static_cast<int>(i), products[i]);
    }
  }

  world_.barrier();
  return true;
}

void StrassenAll::StrassenWorker(int prod_idx, const std::vector<double> &a, const std::vector<double> &b, int size,
                                 std::vector<double> &result, std::mutex &mtx) {
  std::vector<double> local_result;
  if (size <= 32) {
    local_result = StandardMultiply(a, b, size);
  } else {
    int half_size = size / 2;
    std::vector<double> a11(half_size * half_size);
    std::vector<double> a12(half_size * half_size);
    std::vector<double> a21(half_size * half_size);
    std::vector<double> a22(half_size * half_size);
    std::vector<double> b11(half_size * half_size);
    std::vector<double> b12(half_size * half_size);
    std::vector<double> b21(half_size * half_size);
    std::vector<double> b22(half_size * half_size);

    SplitMatrix(a, a11, 0, 0, size);
    SplitMatrix(a, a12, 0, half_size, size);
    SplitMatrix(a, a21, half_size, 0, size);
    SplitMatrix(a, a22, half_size, half_size, size);
    SplitMatrix(b, b11, 0, 0, size);
    SplitMatrix(b, b12, 0, half_size, size);
    SplitMatrix(b, b21, half_size, 0, size);
    SplitMatrix(b, b22, half_size, half_size, size);

    std::vector<std::vector<double>> inputs_a(7);
    std::vector<std::vector<double>> inputs_b(7);
    switch (prod_idx) {
      case 0:
        inputs_a[0] = AddMatrices(a11, a22, half_size);
        inputs_b[0] = AddMatrices(b11, b22, half_size);
        break;
      case 1:
        inputs_a[1] = AddMatrices(a21, a22, half_size);
        inputs_b[1] = b11;
        break;
      case 2:
        inputs_a[2] = a11;
        inputs_b[2] = SubtractMatrices(b12, b22, half_size);
        break;
      case 3:
        inputs_a[3] = a22;
        inputs_b[3] = SubtractMatrices(b21, b11, half_size);
        break;
      case 4:
        inputs_a[4] = AddMatrices(a11, a12, half_size);
        inputs_b[4] = b22;
        break;
      case 5:
        inputs_a[5] = SubtractMatrices(a21, a11, half_size);
        inputs_b[5] = AddMatrices(b11, b12, half_size);
        break;
      case 6:
        inputs_a[6] = SubtractMatrices(a12, a22, half_size);
        inputs_b[6] = AddMatrices(b21, b22, half_size);
        break;
      default:
        break;
    }

    local_result.resize(half_size * half_size);
    if (size <= 32) {
      local_result = StandardMultiply(inputs_a[prod_idx], inputs_b[prod_idx], half_size);
    } else {
      std::vector<std::thread> threads;
      size_t num_threads = ppc::util::GetPPCNumThreads();
      threads.reserve(num_threads);
      std::mutex local_mtx;
      for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&inputs_a, &inputs_b, prod_idx, half_size, &local_result, &local_mtx]() {
          auto temp = StrassenMultiply(inputs_a[prod_idx], inputs_b[prod_idx], half_size);
          std::lock_guard<std::mutex> lock(local_mtx);
          local_result = temp;
        });
      }
      for (auto &t : threads) {
        t.join();
      }
    }
  }

  std::lock_guard<std::mutex> lock(mtx);
  result = local_result;
}

bool StrassenAll::PostProcessingImpl() {
  if (world_.rank() == 0) {
    if (original_size_ != matrix_size_) {
      output_matrix_ = TrimMatrixToOriginalSize(output_matrix_, original_size_, matrix_size_);
    }
    auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);
    std::ranges::copy(output_matrix_, out_ptr);
  }
  return true;
}

std::vector<double> StrassenAll::AddMatrices(const std::vector<double> &a, const std::vector<double> &b, int size) {
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::plus<>());
  return result;
}

std::vector<double> StrassenAll::SubtractMatrices(const std::vector<double> &a, const std::vector<double> &b,
                                                  int size) {
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::minus<>());
  return result;
}

std::vector<double> StandardMultiply(const std::vector<double> &a, const std::vector<double> &b, int size) {
  std::vector<double> result(size * size, 0.0);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        result[(i * size) + j] += a[(i * size) + k] * b[(k * size) + j];
      }
    }
  }
  return result;
}

std::vector<double> StrassenAll::PadMatrixToPowerOfTwo(const std::vector<double> &matrix, int original_size) {
  int new_size = 1;
  while (new_size < original_size) {
    new_size *= 2;
  }
  std::vector<double> padded_matrix(new_size * new_size, 0);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * original_size, matrix.begin() + (i + 1) * original_size,
                      padded_matrix.begin() + i * new_size);
  }
  return padded_matrix;
}

std::vector<double> StrassenAll::TrimMatrixToOriginalSize(const std::vector<double> &matrix, int original_size,
                                                          int padded_size) {
  std::vector<double> trimmed_matrix(original_size * original_size);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * padded_size, matrix.begin() + i * padded_size + original_size,
                      trimmed_matrix.begin() + i * original_size);
  }
  return trimmed_matrix;
}

std::vector<double> StrassenAll::StrassenMultiply(const std::vector<double> &a, const std::vector<double> &b,
                                                  int size) {
  if (size <= 32) {
    return StandardMultiply(a, b, size);
  }

  int half_size = size / 2;
  int half_size_squared = half_size * half_size;

  std::vector<double> a11(half_size_squared);
  std::vector<double> a12(half_size_squared);
  std::vector<double> a21(half_size_squared);
  std::vector<double> a22(half_size_squared);
  std::vector<double> b11(half_size_squared);
  std::vector<double> b12(half_size_squared);
  std::vector<double> b21(half_size_squared);
  std::vector<double> b22(half_size_squared);

  SplitMatrix(a, a11, 0, 0, size);
  SplitMatrix(a, a12, 0, half_size, size);
  SplitMatrix(a, a21, half_size, 0, size);
  SplitMatrix(a, a22, half_size, half_size, size);
  SplitMatrix(b, b11, 0, 0, size);
  SplitMatrix(b, b12, 0, half_size, size);
  SplitMatrix(b, b21, half_size, 0, size);
  SplitMatrix(b, b22, half_size, half_size, size);

  std::vector<double> p1 =
      StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size);
  std::vector<double> p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size);
  std::vector<double> p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size);
  std::vector<double> p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size);
  std::vector<double> p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size);
  std::vector<double> p6 =
      StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size);
  std::vector<double> p7 =
      StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size);

  std::vector<double> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  std::vector<double> c12 = AddMatrices(p3, p5, half_size);
  std::vector<double> c21 = AddMatrices(p2, p4, half_size);
  std::vector<double> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

  std::vector<double> result(size * size);
  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);

  return result;
}

void StrassenAll::SplitMatrix(const std::vector<double> &parent, std::vector<double> &child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(parent.begin() + (row_start + i) * parent_size + col_start,
                      parent.begin() + (row_start + i) * parent_size + col_start + child_size,
                      child.begin() + i * child_size);
  }
}

void StrassenAll::MergeMatrix(std::vector<double> &parent, const std::vector<double> &child, int row_start,
                              int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(child.begin() + i * child_size, child.begin() + (i + 1) * child_size,
                      parent.begin() + (row_start + i) * parent_size + col_start);
  }
}

}  // namespace nasedkin_e_strassen_algorithm_all