// Copyright 2019 Kats Eugeniy

#include <gtest/gtest.h>
#include <mpi.h>

#include <gtest-mpi-listener.hpp>
#include <vector>

#include "./matrix_max.h"

TEST(MATRIX_MAX_PARALL, MATRIX_SIZE_4_5) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> matrix;
  if (rank == 0) {
    matrix = randMatrix(4, 5);
  }

  int actualMax = matrix_max(matrix, 4, 5);
  if (rank == 0) {
    int expMax = *std::max_element(matrix.begin(), matrix.end());
    ASSERT_EQ(expMax, actualMax);
  }
}

TEST(MATRIX_MAX_PARALL, MATRIX_SIZE_10_20) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> matrix;
  if (rank == 0) {
    matrix = randMatrix(10, 20);
  }

  int actualMax = matrix_max(matrix, 10, 20);
  if (rank == 0) {
    int expMax = *std::max_element(matrix.begin(), matrix.end());
    ASSERT_EQ(expMax, actualMax);
  }
}

TEST(MATRIX_MAX_PARALL, MATRIX_SIZE_20_20) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> matrix;
  if (rank == 0) {
    matrix = randMatrix(20, 20);
  }

  int actualMax = matrix_max(matrix, 20, 20);
  if (rank == 0) {
    int expMax = *std::max_element(matrix.begin(), matrix.end());
    ASSERT_EQ(expMax, actualMax);
  }
}

TEST(MATRIX_MAX_PARALL, MATRIX_SIZE_100_10) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> matrix;
  if (rank == 0) {
    matrix = randMatrix(100, 10);
  }

  int actualMax = matrix_max(matrix, 100, 10);
  if (rank == 0) {
    int expMax = *std::max_element(matrix.begin(), matrix.end());
    ASSERT_EQ(expMax, actualMax);
  }
}

TEST(MATRIX_MAX_PARALL, MATRIX_SIZE_100_100) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> matrix;
  if (rank == 0) {
    matrix = randMatrix(100, 100);
  }

  int actualMax = matrix_max(matrix, 100, 100);
  if (rank == 0) {
    int expMax = *std::max_element(matrix.begin(), matrix.end());
    ASSERT_EQ(expMax, actualMax);
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
