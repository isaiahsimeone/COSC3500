#include <string>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <iostream>
#include <fstream>
#include <omp.h>

#include <immintrin.h>

#include "eigensolver.h"
#include "randutil.h"

#define FILE_DUMP 0 // Should this program write results to a file?

using namespace std;

/* Global variables to store the matrix */
double* M = nullptr;
int N = 0;
// Output file to check validity of results
ofstream out_openmp;

void MatrixVectorMultiply(double* Y, const double* X) {
   // Do OpenMP trix here

   memset(Y, 0L, sizeof(double) * N);

   #pragma omp parallel for
   for (int i = 0; i < N; ++i)
   {
      //cout << "Hello, this is thread #" << omp_get_thread_num() << endl;
      for (int j = 0; j < N; ++j)
      {
         Y[i] += M[i*N+j] * X[j];
         //std::cout << "y += " << M[i*N+j] << " * "<< X[j] << std::endl;
      }
      //Y[i] = y;
   }

   // Export Y for validity testing
   #if FILE_DUMP
   for (int i = 0; i < N; i++)
      out_openmp << Y[i] << " ";
   out_openmp << "\n";
   #endif
}

int main(int argc, char** argv) {
   // Output file
   #if FILE_DUMP
   out_openmp.open("openmp_results.txt", ios::out);
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

   //print_matrix(M);

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

   auto FinishTime = std::chrono::high_resolution_clock::now();

   // Close file
   #if FILE_DUMP
   out_openmp.close();
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
   std::cout << "Total serial (initialization + solver): " << std::setw(12) << (TotalTime - Info.TimeInMultiply).count() << " us\n";
   std::cout << "Number of matrix-vector multiplies:     " << std::setw(12) << Info.NumMultiplies << '\n';
   std::cout << "Time per matrix-vector multiplication:  " << std::setw(12) << (Info.TimeInMultiply / Info.NumMultiplies).count() << " us\n";

   // free memory
   free(M);
}
