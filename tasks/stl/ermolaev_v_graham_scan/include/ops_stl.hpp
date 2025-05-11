#pragma once

#include <algorithm>
#include <cmath>
#include <compare>
#include <cstddef>
#include <iterator>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

namespace ermolaev_v_graham_scan_stl {

constexpr size_t kMinInputPoints = 3;
constexpr size_t kMinStackPoints = 2;

class Point {
 public:
  int x;
  int y;

  Point(int x_value, int y_value) : x(x_value), y(y_value) {}
  Point() : x(0), y(0) {}
  bool operator==(const Point& rhs) const { return y == rhs.y && x == rhs.x; }
  bool operator!=(const Point& rhs) const { return !(*this == rhs); }
  auto operator<=>(const Point& rhs) const {
    if (auto cmp = y <=> rhs.y; cmp != 0) {
      return cmp;
    }
    return x <=> rhs.x;
  }
};

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Point> input_, output_;

  static inline int CrossProduct(const Point& p1, const Point& p2, const Point& p3);
  bool CheckGrahamNecessaryConditions();
  void GrahamScan();
  bool IsAllCollinear();
  bool IsAllSame();

  template <typename Iterator, typename Comparator>
  void ParallelSort(Iterator begin, Iterator end, Comparator comp);
};

template <typename Iterator, typename Comparator>
void ermolaev_v_graham_scan_stl::TestTaskSTL::ParallelSort(Iterator begin, Iterator end, Comparator comp) {
  const size_t n = std::distance(begin, end);
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads;
  std::vector<size_t> chunk_offsets(num_threads + 1);

  {
    int i = 0;
    std::transform(chunk_offsets.begin(), chunk_offsets.end(), chunk_offsets.begin(),
                   [&](size_t _) { return ((i++) * n) / num_threads; });
  }

  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back([&, i]() { std::sort(begin + chunk_offsets[i], begin + chunk_offsets[i + 1], comp); });
  }

  for (auto& t : threads) {
    t.join();
  }

  std::vector<typename std::iterator_traits<Iterator>::value_type> buffer(n);

  auto buffer_begin = buffer.begin();
  std::copy(begin, begin + chunk_offsets[1], buffer_begin);

  for (int i = 1; i < num_threads; i++) {
    auto left_begin = buffer_begin;
    auto left_end = buffer_begin + chunk_offsets[i] - chunk_offsets[0];
    auto right_begin = begin + chunk_offsets[i];
    auto right_end = begin + chunk_offsets[i + 1];

    std::merge(left_begin, left_end, right_begin, right_end, begin, comp);
    if (i < num_threads - 1) {
      std::copy(begin, begin + chunk_offsets[i + 1], buffer_begin);
    }
  }
}

}  // namespace ermolaev_v_graham_scan_stl
