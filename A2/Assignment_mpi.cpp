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

         int microsecond = 1000000; // This can go

#define FILE_DUMP             1 // Should this program write results to a file?
#define PROC_ROOT             0
#define SIGNAL_SUBTASK_READY  0xbeef
#define SIGNAL_DIE            0xd1e
#define SIGNAL_NOTHING        0x1

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


/*
 * Send a 'signal' to all slaves.
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
   double* sub_M = new double[N * N / slave_count]{};
   double* my_X = new double[N]{};
   double* my_Y = new double[N / slave_count]{};

   /* Get slave sub_M from root process */
   MPI_Recv(sub_M, N * N / slave_count, MPI_DOUBLE, PROC_ROOT, TAG_DISTRIBUTE_M, MPI_COMM_WORLD, nullptr);

   printf("I (%d) am ready.\n", my_rank);

   while (await_signal() != SIGNAL_DIE) {
      //printf("I (%d) have work available\n", my_rank);
      
      /* Get X */
      MPI_Recv(my_X, N, MPI_DOUBLE, PROC_ROOT, TAG_ACQUIRE_X, MPI_COMM_WORLD, nullptr);

      /* Do the work */
      for (int i = 0; i < N / slave_count; i++) {
         double y = 0;
         for (int j = 0; j < N; j++) {
            y += sub_M[i * N + j] * my_X[j];
         }
         my_Y[i] = y;
      }
   
      /* Send the results back */
      MPI_Send(my_Y, N / slave_count, MPI_DOUBLE, PROC_ROOT, TAG_GET_RESULTS, MPI_COMM_WORLD);
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

   // This cannot go here. The MPI_Send() call's above 
   // Block for N > 505. I think this is because some sort
   // of internal MPI buffer is being filled. Instead,
   // slaves need to be signaled before sending signaling work.
   // This way, each slave will be able to read messages (thus
   // emptying this buffer (if it even exists?))
   //slave_signal(SIGNAL_SUBTASK_READY);

   /* Get results from slaves */
   for (int i = 1; i < world_size; i++)
      MPI_Recv(Y + (N/slave_count) * (i-1), (N/slave_count), MPI_DOUBLE, i, TAG_GET_RESULTS, MPI_COMM_WORLD, nullptr);
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

   /* Only root process from here. The slaves just await jobs */
   if (my_rank == PROC_ROOT) {
      // Output file
      #if FILE_DUMP
      out_mpi.open("mpi_results.txt", ios::out );
      #endif

      if (N % slave_count != 0) {
         std::cerr << "Invalid number of MPI processes\n";
         return 1;
      }
      
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
      for (int i = 1; i < world_size; i++, offset += N * N / slave_count)
         MPI_Send(M + offset, N * N / slave_count, MPI_DOUBLE, i, TAG_DISTRIBUTE_M, MPI_COMM_WORLD);

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