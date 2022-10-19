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
    {"graphical", no_argument, 0, 'g'},
    {"dump-information", no_argument, 0, 'd'},
    {"iterations", required_argument, 0, 'i'},
    {"thread-count", required_argument, 0, 'c'},
    {"lattice-divisions", required_argument, 0, 'l'},
    {"seed", required_argument, 0, 's'},
    {"temperature", required_argument, 0, 't'},
    {"output", required_argument, 0, 'o'}
};

/* Prototypes */
void        usage(char*);
void        print_progress(double, long, long, long);

inline void CUDACheckErr(cudaError_t e) {
   if (e != cudaSuccess) {
      std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
      abort();
   }
}

inline void CUDACheckErr(curandStatus_t e) {
    if (e != CURAND_STATUS_SUCCESS) {
        std::cerr << "CURAND error: " << int(e) << "\n";
        abort();
    }
}

#endif /*__PARALLEL_H_*/
