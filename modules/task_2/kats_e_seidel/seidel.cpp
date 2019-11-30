// Copyright 2019 Kats Eugeniy

#include "../../../modules/task_2/kats_e_seidel/seidel.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>

std::vector<double> randMatrix(int n, int type) {
  if (n < 0) {
    throw - 2;
  }

  std::srand(time(NULL));
  std::vector<double> result;

  if (type == TYPE_A) {
    for (int i = 0; i < n; i++) {
      int sum = 0;
      for (int j = 0; j < n; j++) {
        if (i == j) {
          result.push_back(0);
        } else {
          int temp = std::rand() % 50;
          sum += temp;
          result.push_back(temp);
        }
      }

      result[i * n + i] = sum + 1;
    }
  } else if (type == TYPE_B) {
    for (int i = 0; i < n; i++) {
      result.push_back(std::rand() % 50);
    }
  }

  return result;
}

bool norm(std::vector<double> x, std::vector<double> p, double eps) {
  double norm = 0.0;

  for (int i = 0; i < (signed)x.size(); i++) {
    norm += std::pow(x[i] - p[i], 2);
  }

  return std::sqrt(norm) < eps;
}

std::vector<double> seidel_solve(std::vector<double> A, std::vector<double> B,
                                 int n, double eps) {
  if (n < 0) {
    throw - 2;
  }

  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int delta = n / size;
  int leftBound = delta * rank;
  int rightBound = delta * (rank + 1);

  if (rank == size - 1 && delta != 0) {
    rightBound = n;
  }
  if (rank == 0 && delta == 0) {
    rightBound = n;
  }

  double localSum = 0, globalSum = 0;

  std::vector<double> x(n);
  std::vector<double> p(n);

  bool check;
  do {
    for (int i = 0; i < n; i++) {
      p[i] = x[i];
    }

    for (int i = 0; i < n; i++) {
      localSum = 0.0;
      globalSum = 0.0;

      for (int j = 0; j < i; j++) {
        if ((leftBound <= j) && (j < rightBound)) {
          localSum += (A[i * n + j] * x[j]);
        }
      }

      for (int j = i + 1; j < n; j++) {
        if ((leftBound <= j) && (j < rightBound)) {
          localSum += (A[i * n + j] * p[j]);
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0,
                 MPI_COMM_WORLD);

      if (rank == 0) {
        x[i] = (B[i] - globalSum) / A[i * n + i];
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&x[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      check = !norm(x, p, eps);
    }
    MPI_Bcast(&check, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  } while (check);

  return x;
}

std::vector<double> seidel_solve_s(std::vector<double> A, std::vector<double> B,
                                   int n, double eps) {
  if (n < 0) {
    throw - 2;
  }

  std::vector<double> x(n);
  std::vector<double> p(n);

  do {
    for (int i = 0; i < n; i++) {
      p[i] = x[i];
    }

    for (int i = 0; i < n; i++) {
      double var = 0;

      for (int j = 0; j < i; j++) {
        var += (A[i * n + j] * x[j]);
      }

      for (int j = i + 1; j < n; j++) {
        var += (A[i * n + j] * p[j]);
      }

      x[i] = (B[i] - var) / A[i * n + i];
    }
  } while (!norm(x, p, eps));

  return x;
}
