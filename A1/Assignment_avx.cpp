#include <string>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <iostream>
#include <fstream>

#include <immintrin.h>

#include "eigensolver.h"
#include "randutil.h"
using namespace std;

/* Global variables to store the matrix */
double* M = nullptr;
int N = 0;
// For outputting file
ofstream out_avx;

void MatrixVectorMultiply(double* Y, const double* X) {
   // Do AVX trix here
 
   memset(Y, 0L, sizeof(double) * N);

   asm volatile("#---- AVX INSTRUCTIONS START HERE ----");

   for (int j = 0, i = 0; i < N; i++) {
      /* We want to multiply the matrix columns by the same
       * scalar value. Adapted from https://stackoverflow.com/a/9080351 */
      __m256d scalar = _mm256_set1_pd(X[i]);

       for (j = 0; j + 4 < N; j += 4) {
         /* Get 4 doubles from the matrix column */
         __m256d matcolumn = *((__m256d*)(M + i * N + j));

         /* Multiply matrix column by scalar (vector element) */
         // Product = M[i*N+j] * X[i] four doubles at a time
         __m256d product = _mm256_mul_pd(matcolumn, scalar);

         /* Store the result in Y. Four doubles from index j will be populated */
         // Result = Y[j] + product
         __m256d result = _mm256_add_pd(_mm256_load_pd(Y + j), product);

         // Y[j] += result
         _mm256_storeu_pd(Y + j, result);
      }

      asm volatile("#---- AVX INSTRUCTIONS END HERE ----");

      /* If N % 4 != 0, there will be left over's. We just do this serially */
      while (j < N) {
          Y[j] += M[i * N + j] * X[i];
          j++;
      }
    }


   /*
      for (int i = 0; i < N; ++i)
      {
      double y = 0;
         for (int j = 0; j < N; ++j)
         {
            y += M[i*N+j] * X[j];
            std::cout << "y += " << M[i*N+j] << " * "<< X[j] << std::endl;
         }
         Y[i] = y;
      }
   */

   // Export Y for validity testing
   for (int i = 0; i < N; i++)
      out_avx << Y[i] << " ";
   out_avx << "\n";
   cout << "done" << endl;

}

int main(int argc, char** argv) {
   // Output file
   out_avx.open("avx_results.txt", ios::out);

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

   auto FinishTime = std::chrono::high_resolution_clock::now();


   // Close file
   out_avx.close();

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
