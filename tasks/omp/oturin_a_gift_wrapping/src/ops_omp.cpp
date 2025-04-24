#include "omp/oturin_a_gift_wrapping/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <vector>

double oturin_a_gift_wrapping_omp::ABTP(Coord a, Coord b, Coord c) {
  Coord ab = {.x = b.x - a.x, .y = b.y - a.y};
  Coord cb = {.x = b.x - c.x, .y = b.y - c.y};
  double dot = ((ab.x * cb.x) + (ab.y * cb.y));
  double cross = ((ab.x * cb.y) - (ab.y * cb.x));
  return fabs(atan2(cross, dot));
}

double oturin_a_gift_wrapping_omp::ABTP(Coord a, Coord c) {
  Coord b{.x = a.x, .y = (a.y - 1)};
  return ABTP(b, a, c);
}

double oturin_a_gift_wrapping_omp::Distance(Coord a, Coord b) {
  int t1 = a.x - b.x;
  int t2 = a.y - b.y;
  return sqrt((t1 * t1) + (t2 * t2));
}

bool oturin_a_gift_wrapping_omp::TestTaskOMP::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<Coord *>(task_data->inputs[0]);
  input_ = std::vector<Coord>(in_ptr, in_ptr + input_size);
  n_ = int(input_.size());
  output_ = std::vector<Coord>(0);
  output_.reserve(n_);

  // check if all points are same
  auto are_same = [&](const auto &p) { return p == input_[0]; };
  return !std::ranges::all_of(input_.begin(), input_.end(), are_same);
}

bool oturin_a_gift_wrapping_omp::TestTaskOMP::ValidationImpl() {
  return task_data->inputs_count[0] >= 3;  // task requires 3 or more points to wrap
}

bool oturin_a_gift_wrapping_omp::TestTaskOMP::RunImpl() {
  if (!output_.empty()) {
    output_.clear();
  }
  // this .clear() used ONLY for perftest TaskRun. for some reason output_ has something in it

  // find most left point (priority to top)
  int start_index = FindMostLeft();
  output_.push_back(input_[start_index]);

  // find second point
  double line_angle = -5;
  int search_index = 0;
  for (int i = 0; i < n_; i++) {
    double t = ABTP(input_[start_index], input_[i]);
    if (t > line_angle && i != start_index) {
      line_angle = t;
      search_index = i;
    } else if (t == line_angle) {
      if (Distance(input_[start_index], input_[i]) < Distance(input_[start_index], input_[search_index]) &&
          i != start_index) {
        search_index = i;
        line_angle = t;
      }
    }
  }

  // main loop
  do {
    output_.push_back(input_[search_index]);
    line_angle = -4;
#pragma omp parallel
    {
      int search_index_par = -1;
      double line_angle_private = -5;

      const Coord penultimate_element = output_[output_.size() - 2];
      const Coord last_element = output_.back();

#pragma omp for nowait
      for (int i = 0; i < n_; i++) {
        double t = ABTP(penultimate_element, last_element, input_[i]);
        PointSearch(t, line_angle_private, search_index_par, i);
      }

#pragma omp critical
      {
        if (line_angle <= line_angle_private &&
            (line_angle_private != line_angle ||
             Distance(output_.back(), input_[search_index_par]) < Distance(output_.back(), input_[search_index]))) {
          search_index = search_index_par;
          line_angle = line_angle_private;  // NOLINT(clang-analyzer-deadcode.DeadStores): line_angle will be read in
                                            // the next critical iteration
        }
      }
    }
  } while (search_index != start_index);

  return true;
}

bool oturin_a_gift_wrapping_omp::TestTaskOMP::PostProcessingImpl() {
  auto *result_ptr = reinterpret_cast<Coord *>(task_data->outputs[0]);
  std::ranges::copy(output_.begin(), output_.end(), result_ptr);
  return true;
}

int oturin_a_gift_wrapping_omp::TestTaskOMP::FindMostLeft() {
  Coord most_left = input_[0];
  int start_index = 0;
  for (int i = 1; i < n_; i++) {
    if (input_[i].x < most_left.x || (input_[i].x == most_left.x && input_[i].y > most_left.y)) {
      start_index = i;
      most_left = input_[i];
    }
  }
  return start_index;
}

void oturin_a_gift_wrapping_omp::TestTaskOMP::PointSearch(const double t, double &line_angle, int &search_index,
                                                          const int i) {
  if (t < line_angle) {
    return;
  }
  if (output_.back() != input_[i] && output_[output_.size() - 2] != input_[i]) {
    if (t == line_angle && Distance(output_.back(), input_[i]) >= Distance(output_.back(), input_[search_index])) {
      return;
    }
    search_index = i;
    line_angle = t;
  }
}
