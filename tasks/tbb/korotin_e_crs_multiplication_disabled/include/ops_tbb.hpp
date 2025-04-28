#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace korotin_e_crs_multiplication_tbb {

class CrsMultiplicationTBB : public ppc::core::Task {
 public:
  explicit CrsMultiplicationTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void MulTask(size_t l, size_t r, std::vector<double> &local_val, std::vector<unsigned int> &local_col,
               std::vector<unsigned int> &temp_r_i, const std::vector<unsigned int> &tr_i,
               const std::vector<unsigned int> &tcol, const std::vector<double> &tval);

  std::vector<double> A_val_, B_val_, output_val_;
  std::vector<unsigned int> A_col_, A_rI_, B_col_, B_rI_, output_col_, output_rI_;
  unsigned int A_N_, A_Nz_, B_N_, B_Nz_;
};

std::vector<double> GetRandomMatrix(unsigned int m, unsigned int n);

void MakeCRS(std::vector<unsigned int> &r_i, std::vector<unsigned int> &col, std::vector<double> &val,
             const std::vector<double> &src, unsigned int m, unsigned int n);

void MatrixMultiplication(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c,
                          unsigned int m, unsigned int n, unsigned int p);

}  // namespace korotin_e_crs_multiplication_tbb
