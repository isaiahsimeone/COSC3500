// COSC3500, Semester 2, 2022
// Main file - mpi version

#include "eigensolver.h"
#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <assert.h>

#define FILE_DUMP             0 // Should this program write results to a file?
#define PROC_ROOT             0

#define SIGNAL_SUBTASK_READY  0xbeef
#define SIGNAL_DIE            0xd1e

#define TAG_SIGNAL            0x2
#define TAG_GET_RESULTS       0x3
#define TAG_DISTRIBUTE_M      0x4
#define TAG_ACQUIRE_X         0x5

using namespace std;

// global variables to store the matrix

ofstream out_mpi;
double* M = nullptr;
int N = 0;

int world_size;
int slave_count;
int my_rank;

int rows_per_slave;
int elements_per_slave;

/*
 * Send the specified 'signal' to all slaves.
 */
void slave_signal(int signal) {
   for (int i = 1; i < world_size; i++)
      MPI_Send(&signal, 1, MPI_INT, i, TAG_SIGNAL, MPI_COMM_WORLD);
}

/*
 * Blocks until a signal is received by the root process
 */
int await_signal() {
   int signal = 0;
   MPI_Recv(&signal, 1, MPI_INT, PROC_ROOT, TAG_SIGNAL, MPI_COMM_WORLD, nullptr);
   return signal;
}

/*
 * This function is run by all non-root MPI processes.
 * Each of these processes are allocated a segment of
 * the matrix M. Once the root process has a job available
 * (See MatrixVectorMultiply()), MPI slaves will become active,
 * receive X and compute the matrix vector multiplication for
 * their respective segement of Matrix M
 *
 * Exits only on exception or when slave_signal(SIGNAL_DIE) is called.
 */
void subtask_servicer() {
   double* sub_M = new double[elements_per_slave]{};
   double* my_X = new double[N]{};
   double* my_Y = new double[rows_per_slave]{};

   /* Get slave sub_M from root process */
   MPI_Recv(sub_M, elements_per_slave, MPI_DOUBLE, PROC_ROOT, TAG_DISTRIBUTE_M, MPI_COMM_WORLD, nullptr);

   while (await_signal() != SIGNAL_DIE) {
      //printf("I (%d) have work available\n", my_rank);
      
      /* Get X */
      MPI_Recv(my_X, N, MPI_DOUBLE, PROC_ROOT, TAG_ACQUIRE_X, MPI_COMM_WORLD, nullptr);

      /* Do the work, put the result in my_Y */
      for (int i = 0; i < rows_per_slave; i++) {
         double y = 0;
         for (int j = 0; j < N; j++) {
            y += sub_M[i * N + j] * my_X[j];
         }
         my_Y[i] = y;
      }
   
      /* Send the results back */
      MPI_Send(my_Y, rows_per_slave, MPI_DOUBLE, PROC_ROOT, TAG_GET_RESULTS, MPI_COMM_WORLD);
   }
}

// implementation of the matrix-vector multiply function
// Only the root process can ever run this
void MatrixVectorMultiply(double* Y, const double* X)
{
   /* Tell the slaves that work is ready */
   slave_signal(SIGNAL_SUBTASK_READY);
   
   /* Give X to the slaves */
   for (int i = 1; i < world_size; i++)
      MPI_Send(X, N, MPI_DOUBLE, i, TAG_ACQUIRE_X, MPI_COMM_WORLD);

   /* 
    * Are there rows left over that no slaves will service? 
    * There will be when N % slave_count != 0. The root process
    * can take care of those last few rows while waiting for the 
    * slaves to finish
    */
   int remaining_rows = N - rows_per_slave * slave_count;
   int row_offset = rows_per_slave * slave_count;

   for (int i = 0; i < remaining_rows; i++) {
      double y = 0;
      for (int j = 0; j < N; j++) {
         y += M[N * (row_offset + i) + j] * X[j];
      }
      Y[row_offset + i] = y;
   }

   /* Merge results from the slaves into Y */
   for (int i = 1; i < world_size; i++)
      MPI_Recv(Y + rows_per_slave * (i - 1), rows_per_slave, MPI_DOUBLE, i, TAG_GET_RESULTS, MPI_COMM_WORLD, nullptr);
}

int main(int argc, char** argv)
{

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   
   slave_count = world_size - 1;

   // get the input size from the command line
   if (argc < 2)
   {
      std::cerr << "expected: matrix size <N>\n";
      return 1;
   }
   N = std::stoi(argv[1]);

   rows_per_slave = floor(N / slave_count);
   elements_per_slave = N * rows_per_slave;

   if (rows_per_slave * slave_count != N && my_rank == 0) {
      printf("There will be %d left over rows which root will calculate.\n", N - rows_per_slave * slave_count);
   }

   /* Only root process from here. The slaves just await jobs */
   if (my_rank == PROC_ROOT) {
      // Output file
      #if FILE_DUMP
      out_mpi.open("mpi_results.txt", ios::out );
      #endif
      
      // get the current time, for benchmarking
      auto StartTime = std::chrono::high_resolution_clock::now();

      // Allocate memory for the matrix
      M = static_cast<double*>(malloc(N*N*sizeof(double)));
      
      // seed the random number generator to a known state
      randutil::seed(4);  // The standard random number.  https://xkcd.com/221/

      // Initialize the matrix.  This is a matrix from a Gaussian Orthogonal Ensemble.
      // The matrix is symmetric.
      // The diagonal entries are gaussian distributed with variance 2.
      // The off-diagonal entries are gaussian distributed with variance 1.
      for (int i = 0; i < N; ++i)
      {
         M[i*N+i] = std::sqrt(2.0) * randutil::randn();
         for (int j = i+1; j < N; ++j)
         {
            M[i*N + j] = M[j*N + i] = randutil::randn();
         }
      }


      /* Divide up M for the slaves */
      int offset = 0;
      // each slave gets N*N/slave_count rows of M
      for (int i = 1; i < world_size; i++, offset += elements_per_slave)
         MPI_Send(M + offset, elements_per_slave, MPI_DOUBLE, i, TAG_DISTRIBUTE_M, MPI_COMM_WORLD);
      
      auto FinishInitialization = std::chrono::high_resolution_clock::now();

      // Call the eigensolver
      EigensolverInfo Info = eigenvalues_arpack(N, 100);
      
      /* All slave processes exit */
      slave_signal(SIGNAL_DIE);

      auto FinishTime = std::chrono::high_resolution_clock::now();

      // Close file
      #if FILE_DUMP
      out_mpi.close();
      #endif

      auto InitializationTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishInitialization - StartTime);
      auto TotalTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishTime - StartTime);
      std::cout << "Obtained " << Info.Eigenvalues.size() << " eigenvalues.\n";
      std::cout << "The largest eigenvalue is: " << std::setw(16) << std::setprecision(12) << Info.Eigenvalues.back() << '\n';
      std::cout << "Total time:                             " << std::setw(12) << TotalTime.count() << " us\n";
      std::cout << "Time spent in initialization:           " << std::setw(12) << InitializationTime.count() << " us\n";
      std::cout << "Time spent in eigensolver:              " << std::setw(12) << Info.TimeInEigensolver.count() << " us\n";
      std::cout << "   Of which the multiply function used: " << std::setw(12) << Info.TimeInMultiply.count() << " us\n";
      std::cout << "   And the eigensolver library used:    " << std::setw(12) << (Info.TimeInEigensolver - Info.TimeInMultiply).count() << " us\n";
      std::cout << "Total mpi (initialization + solver): " << std::setw(12) << (TotalTime - Info.TimeInMultiply).count() << " us\n";
      std::cout << "Number of matrix-vector multiplies:     " << std::setw(12) << Info.NumMultiplies << '\n';
      std::cout << "Time per matrix-vector multiplication:  " << std::setw(12) << (Info.TimeInMultiply / Info.NumMultiplies).count() << " us\n";

      // free memory
      free(M);
   }

   /* slave processes only */
   if (my_rank != PROC_ROOT)
      subtask_servicer();

   MPI_Finalize();
}