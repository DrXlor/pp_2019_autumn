// Copyright 2019 Kats Eugeniy

#include <gtest/gtest.h>
#include <mpi.h>

#include <gtest-mpi-listener.hpp>
#include <vector>

#include "./jarvis.h"

TEST(JARVIS_ALGORITHM, PARLL_TEST) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<Point> set(10);
  if (rank == 0) {
    set = {{0, 0}, {1, 2}, {2, 1},  {4, 2},  {4, 3},
           {6, 0}, {5, 0}, {3, -2}, {2, -3}, {1, -1}};
  }
  std::vector<Point> hull = buildHull(set);

  if (rank == 0) {
    std::vector<Point> exp_hull{{0, 0}, {1, 2}, {4, 3}, {6, 0}, {2, -3}};

    ASSERT_EQ(hull, exp_hull);
  }
}

TEST(JARVIS_ALGORITHM, SEQ_TEST) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    std::vector<Point> set{{0, 0}, {1, 2}, {2, 1},  {4, 2},  {4, 3},
                           {6, 0}, {5, 0}, {3, -2}, {2, -3}, {1, -1}};
    std::vector<Point> hull = buildHull_s(set);
    std::vector<Point> exp_hull{{0, 0}, {1, 2}, {4, 3}, {6, 0}, {2, -3}};

    ASSERT_EQ(hull, exp_hull);
  }
}

TEST(JARVIS_ALGORITHM, RANDOM_SET_100) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<Point> set(100);
  if (rank == 0) {
    set = randomSet(100);
  }

  std::vector<Point> hull = buildHull(set);
  if (rank == 0) {
    std::vector<Point> exp_hull = buildHull_s(set);

    ASSERT_EQ(hull, exp_hull);
  }
}

TEST(JARVIS_ALGORITHM, RANDOM_SET_50) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<Point> set(50);
  if (rank == 0) {
    set = randomSet(50);
  }

  std::vector<Point> hull = buildHull(set);
  if (rank == 0) {
    std::vector<Point> exp_hull = buildHull_s(set);

    ASSERT_EQ(hull, exp_hull);
  }
}

TEST(JARVIS_ALGORITHM, RANDOM_SET_25) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<Point> set(25);
  if (rank == 0) {
    set = randomSet(25);
  }

  std::vector<Point> hull = buildHull(set);
  if (rank == 0) {
    std::vector<Point> exp_hull = buildHull_s(set);

    ASSERT_EQ(hull, exp_hull);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners& listeners =
      ::testing::UnitTest::GetInstance()->listeners();

  listeners.Release(listeners.default_result_printer());
  listeners.Release(listeners.default_xml_generator());

  listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
  return RUN_ALL_TESTS();
}
