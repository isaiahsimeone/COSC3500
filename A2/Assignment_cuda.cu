// COSC3500, Semester 2, 2022
// Main file - CUDA version

#include "eigensolver.h"
#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

#define FILE_DUMP 0 // Should this program write results to a file?

using namespace std;

// global variables to store the matrix
ofstream out_cuda;
double* M = 0;
int N = 0;
double *xDevice, *yDevice, *mDevice = 0;
int block_count, thread_count;

/*
 * Determine whether the parameter is an error. If so,
 * print the error and abort the program.
 * git.science.uq.edu.au/cosc3500/cuda/sumarrays-gpu-v1.cu
 */
void checkError(cudaError_t e)
{
   if (e != cudaSuccess)
   {
      std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
      abort();
   }
}

/*
 * Code that runs on the GPU. given a vector X and some segment of M
 * calculate the vector product sum and put the result in Y
 */
__global__
void CUDAKernel(double* xDevice, double* yDevice, double* mDevice, int N) {

   int idx = blockDim.x  * blockIdx.x + threadIdx.x; //+ 1;

   // A matrix of size N is indexed up to N-1
   if (idx > N-1)
      return ;

   double y = 0;
   // Each CUDA thread deals with a whole matrix row
   for (int i = 0; i < N; ++i)
      y += mDevice[i * N + idx] * xDevice[i];

   yDevice[idx] = y;
}


// implementation of the matrix-vector multiply function
void MatrixVectorMultiply(double* Y, const double* X)
{
   // Copy X to the device
   checkError(cudaMemcpy(xDevice, X, N * sizeof(double), cudaMemcpyHostToDevice));
   // Invoke the CUDA kernel
   CUDAKernel<<<block_count, thread_count>>>(xDevice, yDevice, mDevice, N);
   // checkError(cudaDeviceSynchronize()); // Not required 
   // Copy Y from device
   checkError(cudaMemcpy(Y, yDevice, N * sizeof(double), cudaMemcpyDeviceToHost));

   // Export Y to check validity
   #if FILE_DUMP
   for (int i = 0; i < N; i++)
       out_cuda << Y[i] << " ";
   out_cuda << "\n";
   #endif
}


int main(int argc, char** argv)
{
   // Output file
   #if FILE_DUMP
   out_cuda.open("cuda_results.txt", ios::out );
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

   // Allocate CUDA memory
   printf("Attempting to allocate %ld GB of VRAM\n", sizeof(double) * (N + N + N*N) / 1000000000);
   checkError(cudaMalloc(&xDevice, N * sizeof(double)));
   checkError(cudaMalloc(&yDevice, N * sizeof(double)));
   checkError(cudaMalloc(&mDevice, N * N * sizeof(double)));
   printf("OK\n");

   // Number of threads and blocks. A thread count of 128 performed best.
   thread_count = 16;
   block_count = floor(N / thread_count) + 1;

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

   // Since M does not change, it can be copied to the device now
   checkError(cudaMemcpy(mDevice, M, N * N * sizeof(double), cudaMemcpyHostToDevice));

   auto FinishInitialization = std::chrono::high_resolution_clock::now();

   // Call the eigensolver
   EigensolverInfo Info = eigenvalues_arpack(N, 100);

   auto FinishTime = std::chrono::high_resolution_clock::now();

   // Close file
   #if FILE_DUMP
   out_cuda.close();
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
   std::cout << "Total CUDA (initialization + solver):   " << std::setw(12) << (TotalTime - Info.TimeInMultiply).count() << " us\n";
   std::cout << "Number of matrix-vector multiplies:     " << std::setw(12) << Info.NumMultiplies << '\n';
   std::cout << "Time per matrix-vector multiplication:  " << std::setw(12) << (Info.TimeInMultiply / Info.NumMultiplies).count() << " us\n";

   // free memory
   free(M);
}
