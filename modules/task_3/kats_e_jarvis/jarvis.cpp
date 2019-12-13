// Copyright 2019 Kats Eugeniy

#include "../../../modules/task_3/kats_e_jarvis/jarvis.h"

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <vector>

double cross(const Point& a, const Point& b) {
  return (a.x * b.y) - (b.x * a.y);
}

bool is_counter_clockwise(const Point& p, const Point& i, const Point& q) {
  return cross(i - p, q - p) < 0;
}

Point find_leftmost_point(std::vector<Point>::iterator begin,
                          std::vector<Point>::iterator end, int rank,
                          int size) {
  Point most_point =
      *std::min_element(begin, end, [](Point a, Point b) { return a.x < b.x; });
  std::vector<Point> points(size);
  MPI_Gather(&most_point, 2, MPI_DOUBLE, &points[0], 2, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    most_point = *std::min_element(points.begin(), points.end(),
                                   [](Point a, Point b) { return a.x < b.x; });
  }
  MPI_Bcast(&most_point, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return most_point;
}

std::vector<Point> buildHull(std::vector<Point> m_set) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int delta = m_set.size() / size;
  int rem = m_set.size() % size;

  std::vector<Point> buffer(delta);
  MPI_Scatter(&m_set[rem], delta * 2, MPI_DOUBLE, &buffer[0], delta * 2,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    buffer.insert(buffer.begin(), m_set.begin(), m_set.begin() + rem);
  }

  Point leftmost_point =
      find_leftmost_point(buffer.begin(), buffer.end(), rank, size);

  std::vector<Point> hull;
  std::vector<Point> points(size);
  Point endpoint;
  do {
    hull.push_back(leftmost_point);
    endpoint = m_set.at(0);

    for (auto i = buffer.begin(); i < buffer.end(); i++) {
      if (is_counter_clockwise(hull.back(), *i, endpoint) ||
          endpoint == leftmost_point) {
        endpoint = *i;
      }
    }

    MPI_Gather(&endpoint, 2, MPI_DOUBLE, &points[0], 2, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
      for (auto point : points) {
        if (is_counter_clockwise(hull.back(), point, endpoint) ||
            endpoint == leftmost_point) {
          endpoint = point;
        }
      }
    }

    MPI_Bcast(&endpoint, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    leftmost_point = endpoint;
  } while (endpoint != hull.at(0));

  return hull;
}

std::vector<Point> buildHull_s(std::vector<Point> m_set) {
  Point leftmost_point = *std::min_element(
      m_set.begin(), m_set.end(), [](Point a, Point b) { return a.x < b.x; });

  std::vector<Point> hull;
  Point endpoint;
  do {
    hull.push_back(leftmost_point);
    endpoint = m_set.at(0);
    for (auto i : m_set) {
      if (is_counter_clockwise(hull.back(), i, endpoint) ||
          endpoint == leftmost_point) {
        endpoint = i;
      }
    }

    leftmost_point = endpoint;
  } while (endpoint != hull.at(0));

  return hull;
}

std::vector<Point> randomSet(int point_count) {
  std::srand(std::time(nullptr));
  std::vector<Point> result(point_count);

  std::generate(result.begin(), result.end(), []() {
    return Point{static_cast<double>(std::rand() % 100 - 50),
                 static_cast<double>(std::rand() % 100 - 50)};
  });
  return result;
}
