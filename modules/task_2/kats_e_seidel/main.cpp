// Copyright 2019 Kats Eugeniy

#include <gtest/gtest.h>
#include <mpi.h>

#include <gtest-mpi-listener.hpp>
#include <cmath>
#include <vector>

#include "./seidel.h"

TEST(SEIDEL_METHOD, SLAE_SIZE_4) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> A(4 * 4);
  std::vector<double> B(4);
  if (rank == 0) {
    A = randMatrix(4, TYPE_A);
    B = randMatrix(4, TYPE_B);
  }
  MPI_Bcast(&A[0], 4 * 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&B[0], 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> x(seidel_solve(A, B, 4, 1e-6));

  if (rank == 0) {
    std::vector<double> exp(seidel_solve_s(A, B, 4, 1e-6));

    ASSERT_EQ(norm(x, exp, 1e-5), true);
  }
}

TEST(SEIDEL_METHOD, SLAE_SIZE_20) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> A(20 * 20);
  std::vector<double> B(20);
  if (rank == 0) {
    A = randMatrix(20, TYPE_A);
    B = randMatrix(20, TYPE_B);
  }
  MPI_Bcast(&A[0], 20 * 20, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&B[0], 20, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> x(seidel_solve(A, B, 20, 1e-6));

  if (rank == 0) {
    std::vector<double> exp(seidel_solve_s(A, B, 20, 1e-6));

    ASSERT_EQ(norm(x, exp, 1e-5), true);
  }
}

TEST(SEIDEL_METHOD, SLAE_SIZE_600) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> A(600 * 600);
  std::vector<double> B(600);
  if (rank == 0) {
    A = randMatrix(600, TYPE_A);
    B = randMatrix(600, TYPE_B);
  }
  MPI_Bcast(&A[0], 600 * 600, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&B[0], 600, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> x(seidel_solve(A, B, 600, 1e-6));

  if (rank == 0) {
    std::vector<double> exp(seidel_solve_s(A, B, 600, 1e-6));

    ASSERT_EQ(norm(x, exp, 1e-5), true);
  }
}

TEST(SEIDEL_METHOD, SOLVE_PARLL) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> A{2, 1, 1, 3, 5, 2, 2, 1, 4};
  std::vector<double> B{5, 15, 8};
  std::vector<double> x(seidel_solve(A, B, 3, 1e-6));

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
    std::vector<double> x(seidel_solve_s(A, B, 3, 1e-6));
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
