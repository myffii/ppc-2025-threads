#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm_stl {

std::vector<double> StandardMultiply(const std::vector<double>& a, const std::vector<double>& b, int size);

class ThreadPool {
 public:
  explicit ThreadPool(size_t num_threads);
  ~ThreadPool();

  void Enqueue(std::function<void()> task);
  void Wait();

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;
};

class StrassenStl : public ppc::core::Task {
 public:
  explicit StrassenStl(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static std::vector<double> AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int size);
  static std::vector<double> SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b, int size);
  static void SplitMatrix(const std::vector<double>& parent, std::vector<double>& child, int row_start, int col_start,
                          int parent_size);
  static void MergeMatrix(std::vector<double>& parent, const std::vector<double>& child, int row_start, int col_start,
                          int parent_size);
  static std::vector<double> PadMatrixToPowerOfTwo(const std::vector<double>& matrix, int original_size);
  static std::vector<double> TrimMatrixToOriginalSize(const std::vector<double>& matrix, int original_size,
                                                      int padded_size);
  std::vector<double> StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b, int size);

  std::vector<double> input_matrix_a_, input_matrix_b_;
  std::vector<double> output_matrix_;
  int matrix_size_{};
  int original_size_{};

  std::shared_ptr<ThreadPool> thread_pool_;
};

}  // namespace nasedkin_e_strassen_algorithm_stl
