#pragma once

#include <complex>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace solovev_a_matrix_tbb {

struct MatrixInCcsSparse {
  std::vector<std::complex<double>> val;
  std::vector<int> row;
  std::vector<int> col_p;

  int r_n;
  int c_n;
  int n_z;

  MatrixInCcsSparse(int r_nn = 0, int c_nn = 0, int n_zz = 0) {
    r_n = r_nn;
    c_n = c_nn;
    n_z = n_zz;
    row.resize(n_z);
    col_p.resize(c_n + 1);
    val.resize(n_z);
  }
};

class TBBMatMultCcs : public ppc::core::Task {
 public:
  explicit TBBMatMultCcs(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void ComputeColumnSizes();
  void FillMatrixValues();

  MatrixInCcsSparse *M1_ = nullptr;
  MatrixInCcsSparse *M2_ = nullptr;
  MatrixInCcsSparse *M3_ = nullptr;
};

}  // namespace solovev_a_matrix_tbb
