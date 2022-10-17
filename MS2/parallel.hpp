#ifndef __PARALLEL_H_
#define __PARALLEL_H_
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>        /* exp, ceil */
#include <sys/stat.h>   /* mkdir */
#include <getopt.h>

#include "lattice.hpp"
#include "util.hpp"

#define COLOUR_GREEN        "\x1B[32m"
#define COLOUR_AMBER        "\x1B[33m"
#define COLOUR_RED          "\x1B[31m"
#define COLOUR_RESET        "\x1B[0m"

extern char* optarg;
extern int optind, opterr, optopt;

const char* const short_opts = "dgw:h:r:i:s:t:o:";
const option long_opts[] = {
    {"width", required_argument, 0, 'w'},
    {"height", required_argument, 0, 'h'},
    {"refresh-rate", required_argument, 0, 'r'},
    {"graphical", no_argument, 0, 'g'},
    {"dump-information", no_argument, 0, 'd'},
    {"iterations", required_argument, 0, 'i'},
    {"seed", required_argument, 0, 's'},
    {"temperature", required_argument, 0, 't'},
    {"output", required_argument, 0, 'o'}
};

/* Prototypes */
void        usage(char*);

#endif /*__PARALLEL_H_*/
