// Copyright 2019 Kats Eugeniy

#include <gtest/gtest.h>
#include <mpi.h>

#include <gtest-mpi-listener.hpp>
#include <vector>
#include <cmath>

#include "./seidel.h"

TEST(SEIDEL_METHOD, SLAE_SIZE_4) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> A(randMatrix(4, TYPE_A));
  std::vector<double> B(randMatrix(4, TYPE_B));
  std::vector<double> x(seidel_solve(A, B, 4, .0000001));

  if (rank == 0) {
    std::vector<double> exp(seidel_solve_s(A, B, 4, .0000001));
    bool check = true;
    for (int i = 0; i < 4; i++) {
      std::abs(x[i] - exp[i]) <= .0000001 ? check = true : check = false;
    }
    ASSERT_EQ(check, true);
  }
}

TEST(SEIDEL_METHOD, SLAE_SIZE_20) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> A(randMatrix(20, TYPE_A));
  std::vector<double> B(randMatrix(20, TYPE_B));
  std::vector<double> x(seidel_solve(A, B, 20, .0000001));

  if (rank == 0) {
    std::vector<double> exp(seidel_solve_s(A, B, 20, .0000001));
    bool check = true;
    for (int i = 0; i < 20; i++) {
      std::abs(x[i] - exp[i]) <= .0000001 ? check = true : check = false;
    }
    ASSERT_EQ(check, true);
  }
}

TEST(SEIDEL_METHOD, SLAE_SIZE_600) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> A(randMatrix(600, TYPE_A));
  std::vector<double> B(randMatrix(600, TYPE_B));
  std::vector<double> x(seidel_solve(A, B, 600, .0000001));

  if (rank == 0) {
    std::vector<double> exp(seidel_solve_s(A, B, 600, .0000001));
    bool check = true;
    for (int i = 0; i < 600; i++) {
      std::abs(x[i] - exp[i]) <= .0000001 ? check = true : check = false;
    }
    ASSERT_EQ(check, true);
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
