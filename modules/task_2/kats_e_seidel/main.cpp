// Copyright 2019 Kats Eugeniy

#include <gtest/gtest.h>
#include <mpi.h>

#include <gtest-mpi-listener.hpp>
#include <vector>
#include <cmath>

#include "./seidel.h"

TEST(SEIDEL_METHOD, SLAE_SIZE_3) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size = 3;

  std::vector<double> A;
  std::vector<double> B;

  if (rank == 0) {
    A = randMatrix(size, TYPE_A);
    B = randMatrix(size, TYPE_B);
  } else {
    A.resize(size * size);
    B.resize(size);
  }
  MPI_Bcast(&A[0], size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&B[0], size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> x(seidel_solve(A, B, size, .001));

  if (rank == 0) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
      sum += A[i] * x[i];
    }

    ASSERT_LE(std::round(sum) - B[0], 0);
  }
}

TEST(SEIDEL_METHOD, SLAE_SIZE_5) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size = 5;

  std::vector<double> A;
  std::vector<double> B;

  if (rank == 0) {
    A = randMatrix(size, TYPE_A);
    B = randMatrix(size, TYPE_B);
  } else {
    A.resize(size * size);
    B.resize(size);
  }
  MPI_Bcast(&A[0], size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&B[0], size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> x(seidel_solve(A, B, size, 1e-7));

  if (rank == 0) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
      sum += A[i] * x[i];
    }

    ASSERT_LE(std::round(sum) - B[0], 0);
  }
}

TEST(SEIDEL_METHOD, SLAE_SIZE_10) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size = 10;

  std::vector<double> A;
  std::vector<double> B;

  if (rank == 0) {
    A = randMatrix(size, TYPE_A);
    B = randMatrix(size, TYPE_B);
  } else {
    A.resize(size * size);
    B.resize(size);
  }
  MPI_Bcast(&A[0], size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&B[0], size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> x(seidel_solve(A, B, size, 1e-7));

  if (rank == 0) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
      sum += A[i] * x[i];
    }

    ASSERT_LE(std::round(sum) - B[0], 0);
  }
}

TEST(SEIDEL_METHOD, SLAE_SIZE_50) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size = 50;

  std::vector<double> A;
  std::vector<double> B;

  if (rank == 0) {
    A = randMatrix(size, TYPE_A);
    B = randMatrix(size, TYPE_B);
  } else {
    A.resize(size * size);
    B.resize(size);
  }
  MPI_Bcast(&A[0], size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&B[0], size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> x(seidel_solve(A, B, size, 1e-7));

  if (rank == 0) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
      sum += A[i] * x[i];
    }

    ASSERT_LE(std::round(sum) - B[0], 0);
  }
}

TEST(SEIDEL_METHOD, SOLVE_PARLL) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> A{2, 1, 1, 3, 5, 2, 2, 1, 4};
  std::vector<double> B{5, 15, 8};
  std::vector<double> x(seidel_solve(A, B, 3, .001));

  if (rank == 0) {
    std::vector<double> exp{1, 2, 1};
    for (auto&& elem : x) {
      elem = std::round(elem);
    }

    ASSERT_EQ(exp, x);
  }
}

TEST(SEIDEL_METHOD, SOLVE_SEQ) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    std::vector<double> A{2, 1, 1, 3, 5, 2, 2, 1, 4};
    std::vector<double> B{5, 15, 8};
    std::vector<double> x(seidel_solve_s(A, B, 3, .001));
    std::vector<double> exp{1, 2, 1};

    for (auto&& elem : x) {
      elem = std::round(elem);
    }

    ASSERT_EQ(exp, x);
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
