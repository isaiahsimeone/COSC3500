// COSC3500, Semester 2, 2022
// Main file - mpi version

#include "eigensolver.h"
#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>

#define FILE_DUMP 1 // Should this program write results to a file?

using namespace std;

// global variables to store the matrix

ofstream out_mpi;
double* M = nullptr;
int N = 0;
int world_size;

// MPI process variables
int my_rank;
double* my_X;

// implementation of the matrix-vector multiply function
void MatrixVectorMultiply(double* Y, const double* X)
{

   MPI_Barrier(MPI_COMM_WORLD); 

   for (int i = 0; i < N; i++) {
      double y = 0;
      for (int j = 0; j < N; j++) {
         y += M[i*N + j] * X[j];
      }
      Y[i] = y;
   }

   MPI_Barrier(MPI_COMM_WORLD);

   // We can split the range of i as we did below
   // Then, we just need to synchronise y with the root proc




   /*
   for (int i = (N/world_size) * my_rank; i < (N/world_size) * (my_rank + 1) - 1; i++) {
      double y = 0;
      for (int j = 0; j < N; j++) {
         y += M[i * N + j] * X[j];
      }
      Y[i] = y;
   }
   */

   // Export Y to check validity
   #if FILE_DUMP
   if (my_rank == 0) {
      for (int i = 0; i < N; i++)
         out_mpi << Y[i] << " ";
      out_mpi << "\n";
   }
   #endif

}

int main(int argc, char** argv)
{

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   // Output file
   #if FILE_DUMP
   if (my_rank == 0)
      out_mpi.open("mpi_results.txt", ios::out );
   #endif

   // get the current time, for benchmarking
   auto StartTime = std::chrono::high_resolution_clock::now();

   // get the input size from the command line
   if (argc < 2)
   {
      std::cerr << "expected: matrix size <N>\n";
      return 1;
   }
   N = std::stoi(argv[1]);

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
 

   auto FinishInitialization = std::chrono::high_resolution_clock::now();

   // Call the eigensolver
   EigensolverInfo Info = eigenvalues_arpack(N, 100);

   MPI_Barrier(MPI_COMM_WORLD);

   auto FinishTime = std::chrono::high_resolution_clock::now();

   // Close file
   #if FILE_DUMP
   if (my_rank == 0)
      out_mpi.close();
   #endif

   auto InitializationTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishInitialization - StartTime);
   auto TotalTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishTime - StartTime);
   std::cout << "Report from proc with rank " << my_rank << "\n"; 
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
      
   MPI_Finalize();
}
