#include "omp/shulpin_i_jarvis_passage/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
int Orientation(const shulpin_i_jarvis_omp::Point& p, const shulpin_i_jarvis_omp::Point& q,
                const shulpin_i_jarvis_omp::Point& r) {
  double val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y));
  if (std::fabs(val) < 1e-9) {
    return 0;
  }
  return (val > 0) ? 1 : 2;
}
}  // namespace

void shulpin_i_jarvis_omp::JarvisSequential::MakeJarvisPassage(std::vector<shulpin_i_jarvis_omp::Point>& input_jar,
                                                               std::vector<shulpin_i_jarvis_omp::Point>& output_jar) {
  size_t total = input_jar.size();
  output_jar.clear();

  size_t start = 0;
  for (size_t i = 1; i < total; ++i) {
    if (input_jar[i].x < input_jar[start].x ||
        (input_jar[i].x == input_jar[start].x && input_jar[i].y < input_jar[start].y)) {
      start = i;
    }
  }

  size_t active = start;
  do {
    output_jar.emplace_back(input_jar[active]);
    size_t candidate = (active + 1) % total;

    for (size_t index = 0; index < total; ++index) {
      if (Orientation(input_jar[active], input_jar[index], input_jar[candidate]) == 2) {
        candidate = index;
      }
    }

    active = candidate;
  } while (active != start);
}

bool shulpin_i_jarvis_omp::JarvisSequential::PreProcessingImpl() {
  std::vector<shulpin_i_jarvis_omp::Point> tmp_input;

  auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_omp::Point*>(task_data->inputs[0]);
  size_t tmp_size = task_data->inputs_count[0];
  tmp_input.assign(tmp_data, tmp_data + tmp_size);

  input_seq_ = tmp_input;

  size_t output_size = task_data->outputs_count[0];
  output_seq_.resize(output_size);

  return true;
}

bool shulpin_i_jarvis_omp::JarvisSequential::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
}

bool shulpin_i_jarvis_omp::JarvisSequential::RunImpl() {
  MakeJarvisPassage(input_seq_, output_seq_);
  return true;
}

bool shulpin_i_jarvis_omp::JarvisSequential::PostProcessingImpl() {
  auto* result = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(output_seq_.begin(), output_seq_.end(), result);
  return true;
}

void shulpin_i_jarvis_omp::JarvisOMPParallel::MakeJarvisPassageOMP(
    std::vector<shulpin_i_jarvis_omp::Point>& input_jar, std::vector<shulpin_i_jarvis_omp::Point>& output_jar) {
  int total = static_cast<int>(input_jar.size());
  output_jar.clear();

  std::unordered_set<shulpin_i_jarvis_omp::Point, shulpin_i_jarvis_omp::PointHash, shulpin_i_jarvis_omp::PointEqual>
      unique_points;
  std::vector<shulpin_i_jarvis_omp::Point> hull;
  hull.reserve(total);

  int start = 0;
  for (int i = 1; i < total; ++i) {
    const auto& a = input_jar[i];
    const auto& b = input_jar[start];
    if (a.x < b.x || (a.x == b.x && a.y < b.y)) {
      start = i;
    }
  }
  int active = start;

  do {
    const Point& current = input_jar[active];
    if (unique_points.find(current) == unique_points.end()) {
      hull.push_back(current);
      unique_points.insert(current);
    }

    int candidate = (active + 1) % total;
    std::vector<int> thread_candidates(omp_get_max_threads(), candidate);

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int local_candidate = candidate;

#pragma omp for nowait
      for (int i = 0; i < total; ++i) {
        if (i == active) continue;
        if (Orientation(current, input_jar[i], input_jar[local_candidate]) == 2) {
          local_candidate = i;
        }
      }

      thread_candidates[tid] = local_candidate;
    }

    for (int tid = 0; tid < static_cast<int>(thread_candidates.size()); ++tid) {
      int cand = thread_candidates[tid];
      if (Orientation(current, input_jar[cand], input_jar[candidate]) == 2) {
        candidate = cand;
      }
    }

    if (candidate == active) {
      break;
    }

    active = candidate;

  } while (active != start);

  output_jar = std::move(hull);
}

bool shulpin_i_jarvis_omp::JarvisOMPParallel::PreProcessingImpl() {
  std::vector<shulpin_i_jarvis_omp::Point> tmp_input;

  auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_omp::Point*>(task_data->inputs[0]);
  size_t tmp_size = task_data->inputs_count[0];
  tmp_input.assign(tmp_data, tmp_data + tmp_size);

  input_omp_ = tmp_input;

  size_t output_size = task_data->outputs_count[0];
  output_omp_.resize(output_size);
  return true;
}

bool shulpin_i_jarvis_omp::JarvisOMPParallel::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
}

bool shulpin_i_jarvis_omp::JarvisOMPParallel::RunImpl() {
  MakeJarvisPassageOMP(input_omp_, output_omp_);
  return true;
}

bool shulpin_i_jarvis_omp::JarvisOMPParallel::PostProcessingImpl() {
  auto* result = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(output_omp_.begin(), output_omp_.end(), result);
  return true;
}
