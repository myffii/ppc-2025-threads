#include "stl/odintsov_m_multmatrix_cannon/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <thread>
#include <vector>

void odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL::ShiftRow(std::vector<double>& matrix, int root, int row,
                                                                   int shift) {
  shift = shift % root;
  std::vector<double> tmp(root);
  for (int j = 0; j < root; j++) {
    tmp[j] = matrix[(row * root) + ((j + shift) % root)];
  }
  for (int j = 0; j < root; j++) {
    matrix[(row * root) + j] = tmp[j];
  }
}

void odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL::ShiftColumn(std::vector<double>& matrix, int root, int col,
                                                                      int shift) {
  shift = shift % root;
  std::vector<double> tmp(root);

  for (int i = 0; i < root; i++) {
    tmp[i] = matrix[(((i + shift) % root) * root) + col];
  }
  for (int i = 0; i < root; i++) {
    matrix[(i * root) + col] = tmp[i];
  }
}
void odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL::ShiftBlocksUp(std::vector<double>& matrix, int root,
                                                                        int sz) const {
  int p = root / block_sz_;
  for (int bj = 0; bj < p; bj++) {
    std::vector<double> first_block(block_sz_ * block_sz_);

    for (int i = 0; i < block_sz_; i++) {
      for (int j = 0; j < block_sz_; j++) {
        first_block[(i * block_sz_) + j] = matrix[(i * root) + ((bj * block_sz_) + j)];
      }
    }

    for (int bi = 0; bi < (p - 1); bi++) {
      for (int i = 0; i < block_sz_; i++) {
        for (int j = 0; j < block_sz_; j++) {
          matrix[((bi * block_sz_ + i) * root) + (bj * block_sz_) + j] =
              matrix[(((bi + 1) * block_sz_) * root) + (i * root) + ((bj * block_sz_) + j)];
        }
      }
    }

    for (int i = 0; i < block_sz_; i++) {
      for (int j = 0; j < block_sz_; j++) {
        matrix[((((p - 1) * block_sz_) * root) + (i * root) + ((bj * block_sz_) + j))] =
            first_block[(i * block_sz_) + j];
      }
    }
  }
}

void odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL::ShiftBlocksLeft(std::vector<double>& matrix, int root,
                                                                          int sz) const {
  int p = root / block_sz_;
  for (int bi = 0; bi < p; bi++) {
    std::vector<double> first_block(block_sz_ * block_sz_);

    for (int i = 0; i < block_sz_; i++) {
      for (int j = 0; j < block_sz_; j++) {
        first_block[(i * block_sz_) + j] = matrix[((bi * block_sz_ + i) * root) + j];
      }
    }

    for (int bj = 0; bj < (p - 1); bj++) {
      for (int i = 0; i < block_sz_; i++) {
        for (int j = 0; j < block_sz_; j++) {
          matrix[((bi * block_sz_ + i) * root) + (bj * block_sz_) + j] =
              matrix[((bi * block_sz_ + i) * root) + (((bj + 1) * block_sz_) + j)];
        }
      }
    }

    for (int i = 0; i < block_sz_; i++) {
      for (int j = 0; j < block_sz_; j++) {
        matrix[((bi * block_sz_ + i) * root) + (((p - 1) * block_sz_) + j)] = first_block[(i * block_sz_) + j];
      }
    }
  }
}

bool odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL::IsSquere(unsigned int num) {
  auto root = static_cast<unsigned int>(std::sqrt(num));
  return (root * root) == num;
}

int odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL::GetBlockSize(int n) {
  for (int k = (n / 2); k >= 2; k--) {
    if ((n % k) == 0) {
      return k;
    }
  }
  return 1;
}
void odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL::CopyBlock(const std::vector<double>& matrix,
                                                                    std::vector<double>& block, int start, int root,
                                                                    int block_sz) {
  for (int i = 0; i < block_sz; i++) {
    for (int j = 0; j < block_sz; j++) {
      int index = start + (i * root) + j;
      block[(i * block_sz) + j] = matrix[index];
    }
  }
}
void odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL::InitializeShift(std::vector<double>& matrix, int root,
                                                                          int grid_size, int block_sz,
                                                                          bool is_row_shift) {
  for (int b = 0; b < grid_size; ++b) {
    for (int index = b * block_sz; index < (b + 1) * block_sz; ++index) {
      for (int shift = 0; shift < b; ++shift) {
        if (is_row_shift) {
          ShiftRow(matrix, root, index, block_sz);
        } else {
          ShiftColumn(matrix, root, index, block_sz);
        }
      }
    }
  }
}
void odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL::ProcessBlock(int bi, int num_blocks, int root, int block_sz,
                                                                       const std::vector<double>& matrix_a,
                                                                       const std::vector<double>& matrix_b,
                                                                       std::vector<double>& local_c) {
  std::vector<std::vector<double>> local_blocks_a(num_blocks, std::vector<double>(block_sz * block_sz));
  std::vector<std::vector<double>> local_blocks_b(num_blocks, std::vector<double>(block_sz * block_sz));

  for (int bj = 0; bj < num_blocks; ++bj) {
    int start = ((bi * block_sz) * root) + (bj * block_sz);
    CopyBlock(matrix_a, local_blocks_a[bj], start, root, block_sz);
    CopyBlock(matrix_b, local_blocks_b[bj], start, root, block_sz);
  }

  for (int bj = 0; bj < num_blocks; ++bj) {
    const auto& a = local_blocks_a[bj];
    const auto& b = local_blocks_b[bj];

    for (int i = 0; i < block_sz; ++i) {
      for (int k = 0; k < block_sz; ++k) {
        double a_ik = a[(i * block_sz) + k];
        for (int j = 0; j < block_sz; ++j) {
          int index = (((bi * block_sz) + i) * root) + ((bj * block_sz) + j);
          local_c[index] += a_ik * b[((k * block_sz) + j)];
        }
      }
    }
  }
}

bool odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL::PreProcessingImpl() {
  szA_ = task_data->inputs_count[0];
  szB_ = task_data->inputs_count[1];
  matrixA_.assign(reinterpret_cast<double*>(task_data->inputs[0]),
                  reinterpret_cast<double*>(task_data->inputs[0]) + szA_);
  matrixB_.assign(reinterpret_cast<double*>(task_data->inputs[1]),
                  reinterpret_cast<double*>(task_data->inputs[1]) + szB_);
  matrixC_.assign(szA_, 0);

  block_sz_ = GetBlockSize(static_cast<int>(sqrt(szA_)));
  return true;
}

bool odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL::ValidationImpl() {
  if (task_data->inputs_count[0] != task_data->inputs_count[1]) {
    return false;
  }

  if ((!(IsSquere(task_data->inputs_count[0]))) || (!(IsSquere(task_data->inputs_count[1])))) {
    return false;
  }
  return true;
}

bool odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL::RunImpl() {
  int root = static_cast<int>(std::sqrt(szA_));
  int num_blocks = std::max(1, root / block_sz_);
  int grid_size = num_blocks;
  const int raw_max = ppc::util::GetPPCNumThreads();
  const int max_threads = std::max(1, raw_max);

  const int threads_limit = std::min(num_blocks, max_threads);

  InitializeShift(matrixA_, root, grid_size, block_sz_, true);
  InitializeShift(matrixB_, root, grid_size, block_sz_, false);

  for (int step = 0; step < grid_size; ++step) {
    for (int offset = 0; offset < num_blocks; offset += threads_limit) {
      int batch_size = std::min(threads_limit, num_blocks - offset);

      std::vector<std::vector<double>> local_results(batch_size, std::vector<double>(root * root, 0.0));
      std::vector<std::thread> threads;
      threads.reserve(batch_size);

      for (int bi = 0; bi < batch_size; ++bi) {
        threads.emplace_back([&, bi]() {
          ProcessBlock(offset + bi, num_blocks, root, block_sz_, matrixA_, matrixB_, local_results[bi]);
        });
      }

      for (auto& t : threads) {
        t.join();
      }
      const size_t matrix_size = matrixC_.size();
      for (const auto& local_c : local_results) {
        for (size_t i = 0; i < matrix_size; ++i) {
          matrixC_[i] += local_c[i];
        }
      }
    }

    ShiftBlocksLeft(matrixA_, root, block_sz_);
    ShiftBlocksUp(matrixB_, root, block_sz_);
  }

  return true;
}

bool odintsov_m_mulmatrix_cannon_stl::MulMatrixCannonSTL::PostProcessingImpl() {
  std::size_t sz_c = matrixC_.size();
  for (std::size_t i = 0; i < sz_c; i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = matrixC_[i];
  }
  return true;
}
