#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/tyshkevich_a_hoare_simple_merge/include/ops_omp.hpp"

namespace {
template <typename T>
std::vector<T> GenRandVec(size_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dist(-1000, 1000);

  std::vector<T> vec(size);
  std::ranges::generate(vec, [&] { return dist(gen); });

  return vec;
}

template <typename T, typename Comparator>
void TestSort(std::vector<T> &&in, Comparator cmp) {
  std::vector<T> out(in.size());

  auto dat = std::make_shared<ppc::core::TaskData>();
  dat->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  dat->inputs_count.emplace_back(in.size());
  dat->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  dat->outputs_count.emplace_back(out.size());

  auto tt = tyshkevich_a_hoare_simple_merge_omp::CreateHoareTestTask<T>(dat, cmp);
  ASSERT_EQ(tt.Validation(), true);
  tt.PreProcessing();
  tt.Run();
  tt.PostProcessing();

  ASSERT_EQ(std::ranges::is_sorted(out, cmp), true);
}

template <typename T, typename Comparator>
void TestSort(std::size_t size, Comparator cmp) {
  TestSort(GenRandVec<T>(size), cmp);
}
}  // namespace

TEST(tyshkevich_a_hoare_simple_merge_omp, test_0_gt) { TestSort<int>(0, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_0_lt) { TestSort<int>(0, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_1_gt) { TestSort<int>(1, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_1_lt) { TestSort<int>(1, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_2_gt) { TestSort<int>(2, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_2_lt) { TestSort<int>(2, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_3_gt) { TestSort<int>(3, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_3_lt) { TestSort<int>(3, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_4_gt) { TestSort<int>(4, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_4_lt) { TestSort<int>(4, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_5_gt) { TestSort<int>(5, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_5_lt) { TestSort<int>(5, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_7_gt) { TestSort<int>(7, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_7_lt) { TestSort<int>(7, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_9_gt) { TestSort<int>(9, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_9_lt) { TestSort<int>(9, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_10_gt) { TestSort<int>(10, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_10_lt) { TestSort<int>(10, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_11_gt) { TestSort<int>(11, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_11_lt) { TestSort<int>(11, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_13_gt) { TestSort<int>(13, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_13_lt) { TestSort<int>(13, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_19_gt) { TestSort<int>(19, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_19_lt) { TestSort<int>(19, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_23_gt) { TestSort<int>(23, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_23_lt) { TestSort<int>(23, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_31_gt) { TestSort<int>(31, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_31_lt) { TestSort<int>(31, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_64_gt) { TestSort<int>(64, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_64_lt) { TestSort<int>(64, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_100_gt) { TestSort<int>(100, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_100_lt) { TestSort<int>(100, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_999_gt) { TestSort<int>(999, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_999_lt) { TestSort<int>(999, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_1025_gt) { TestSort<int>(1025, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_1025_lt) { TestSort<int>(1025, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_omp, test_homogeneous_gt) { TestSort<int>({1, 1, 1}, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_omp, test_homogeneous_lt) { TestSort<int>({1, 1, 1}, std::less<>()); }
