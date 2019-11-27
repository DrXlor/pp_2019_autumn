// Copyright 2019 Kats Eugeniy

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "../../../modules/task_1/kats_e_matrix_max/matrix_max.h"

std::vector<int> randMatrix(int rows, int cols) {
  if (rows < 0 || cols < 0) {
    throw - 2;
  }

  std::srand(time(NULL));

  std::vector<int> result(rows * cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result[i * cols + j] = std::rand() % 100;
    }
  }

  return result;
}

int matrix_max(std::vector<int> matrix, int rows, int cols) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int delta = (rows * cols) / size, rem = (rows * cols) % size;

  std::vector<int> buffer(delta + rem);

  if (rank == 0) {
    for (int proc = 1; proc < size; proc++) {
      MPI_Send(&matrix[proc * delta + rem], delta, MPI_INT, proc, 0,
               MPI_COMM_WORLD);
    }

    buffer = std::vector<int>(matrix.begin(), matrix.begin() + delta + rem);
  } else {
    MPI_Recv(&buffer[0], delta, MPI_INT, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  int localMax = *std::max_element(buffer.begin(), buffer.end());
  int globalMax;

  MPI_Reduce(&localMax, &globalMax, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  return globalMax;
}
