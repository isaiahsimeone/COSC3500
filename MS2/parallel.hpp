#ifndef __PARALLEL_H_
#define __PARALLEL_H_
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>        /* exp, ceil */
#include <sys/stat.h>   /* mkdir */
#include <getopt.h>

/* CUDA */
#include <cuda.h>
#include <curand.h>

#include "lattice.hpp"
#include "util.hpp"

#define PARITY_EVEN 0
#define PARITY_ODD  1

extern char* optarg;
extern int optind, opterr, optopt;

const char* const short_opts = "dgn:r:i:s:t:o:c:";
const option long_opts[] = {
    {"dimension", required_argument, 0, 'n'},
    {"update-rate", required_argument, 0, 'r'},
    {"iterations", required_argument, 0, 'i'},
    {"seed", required_argument, 0, 's'},
    {"temperature", required_argument, 0, 't'},
    {"graphical", no_argument, 0, 'g'},
    {"dump-information", no_argument, 0, 'd'},
    {"output", required_argument, 0, 'o'},
    {"thread-count", required_argument, 0, 'c'}
};

/* Prototypes */
void                    usage(char*);
void                    print_progress(double, long, long, long);
__global__ void         evaluate_disjoint_component(int*, const float*, int, float, int);

/*
 * Check if the specified argument indicates an error with a CUDA function
 */
inline void CUDACheckErr(cudaError_t e) {
   if (e != cudaSuccess) {
      std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
      abort();
   }
}

/*
 * Check if the specified argument indicates an error with a CURand function
 */
inline void CUDACheckErr(curandStatus_t e) {
    if (e != CURAND_STATUS_SUCCESS) {
        std::cerr << "CURAND error: " << int(e) << "\n";
        abort();
    }
}

#endif /*__PARALLEL_H_*/
