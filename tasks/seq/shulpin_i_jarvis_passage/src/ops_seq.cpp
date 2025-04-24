#include "seq/shulpin_i_jarvis_passage/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <unordered_set>
#include <vector>

int shulpin_i_jarvis_seq::JarvisSequential::Orientation(const Point& p, const Point& q, const Point& r) {
  double val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y));
  if (std::fabs(val) < 1e-9) {
    return 0;
  }
  return (val > 0) ? 1 : 2;
}

void shulpin_i_jarvis_seq::JarvisSequential::MakeJarvisPassage(std::vector<shulpin_i_jarvis_seq::Point>& input_jar,
                                                               std::vector<shulpin_i_jarvis_seq::Point>& output_jar) {
  size_t total = input_jar.size();
  output_jar.clear();
  // clang-format off
  std::unordered_set<shulpin_i_jarvis_seq::Point, 
                     shulpin_i_jarvis_seq::PointHash, 
                     shulpin_i_jarvis_seq::PointEqual> unique_points;
  // clang-format on
  size_t start = 0;
  for (size_t i = 1; i < total; ++i) {
    if (input_jar[i].x < input_jar[start].x ||
        (input_jar[i].x == input_jar[start].x && input_jar[i].y < input_jar[start].y)) {
      start = i;
    }
  }

  size_t active = start;
  do {
    const auto& current = input_jar[active];
    if (unique_points.find(current) == unique_points.end()) {
      output_jar.emplace_back(current);
      unique_points.insert(current);
    }

    size_t candidate = (active + 1) % total;
    for (size_t index = 0; index < total; ++index) {
      if (index == active) {
        continue;
      }

      if (Orientation(input_jar[active], input_jar[index], input_jar[candidate]) == 2) {
        candidate = index;
      }
    }

    active = candidate;
  } while (active != start);
}

bool shulpin_i_jarvis_seq::JarvisSequential::PreProcessingImpl() {
  std::vector<shulpin_i_jarvis_seq::Point> tmp_input;

  auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_seq::Point*>(task_data->inputs[0]);
  size_t tmp_size = task_data->inputs_count[0];
  tmp_input.assign(tmp_data, tmp_data + tmp_size);

  input_ = tmp_input;

  size_t output_size = task_data->outputs_count[0];
  output_.resize(output_size);

  return true;
}

bool shulpin_i_jarvis_seq::JarvisSequential::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
}

bool shulpin_i_jarvis_seq::JarvisSequential::RunImpl() {
  MakeJarvisPassage(input_, output_);
  return true;
}

bool shulpin_i_jarvis_seq::JarvisSequential::PostProcessingImpl() {
  int* result = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(reinterpret_cast<int*>(output_.data()), reinterpret_cast<int*>(output_.data() + output_.size()),
                    result);
  return true;
}
