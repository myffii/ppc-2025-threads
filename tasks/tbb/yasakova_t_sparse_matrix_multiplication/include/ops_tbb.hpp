#include <complex>
#include <cstdlib>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using ComplexNumber = std::complex<double>;

namespace yasakova_t_sparse_matrix_multiplication {

struct CompressedRowStorageMatrix {
  std::vector<ComplexNumber> nonZeroValues;
  std::vector<int> columnIndices;
  std::vector<int> rowPointers;
  int rowCount;
  int columnCount;
  CompressedRowStorageMatrix() : nonZeroValues({}), columnIndices({}), rowPointers({}), rowCount(0), columnCount(0) {};
  CompressedRowStorageMatrix(int rows, int cols) : rowCount(rows), columnCount(cols) {
    rowPointers.resize(rows + 1, 0);
  }

  void InsertElement(int row, ComplexNumber value, int col);
  CompressedRowStorageMatrix(const CompressedRowStorageMatrix& other) = default;
  CompressedRowStorageMatrix& operator=(const CompressedRowStorageMatrix& other) = default;
  static void DisplayMatrix(const CompressedRowStorageMatrix& matrix);
};
std::vector<ComplexNumber> ConvertMatrixToVector(const CompressedRowStorageMatrix& mat);
CompressedRowStorageMatrix ConvertVectorToMatrix(std::vector<ComplexNumber>& vec);
bool CompareMatrices(const CompressedRowStorageMatrix& first_matrix, const CompressedRowStorageMatrix& second_matrix);
bool AreEqualElems(const ComplexNumber& first_matrix, const ComplexNumber& second_matrix, double tolerance);
class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<ComplexNumber> inputData_, resultData_;
  CompressedRowStorageMatrix firstMatrix_, secondMatrix_;
};

}  // namespace yasakova_t_sparse_matrix_multiplication
