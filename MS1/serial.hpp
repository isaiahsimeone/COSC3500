#ifndef __SERIAL_H_
#define __SERIAL_H_
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <chrono>
#include <getopt.h>
#include <unistd.h>
#include <assert.h>
#include <sys/stat.h>
#include <fcntl.h>

#define COLOUR_GREEN        "\x1B[32m"
#define COLOUR_RESET        "\x1B[0m"

#define BITMAP_HEADER_SZ    54
#define BYTES_PER_MEGABYTE  1000000

typedef std::chrono::time_point<std::chrono::high_resolution_clock> _time_point;
typedef std::chrono::high_resolution_clock                          _clock;

extern char* optarg;
extern int optind, opterr, optopt;

const char* const short_opts = "gw:h:r:i:s:t:o:";
const option long_opts[] = {
    {"width", required_argument, 0, 'w'},
    {"height", required_argument, 0, 'h'},
    {"refresh-rate", required_argument, 0, 'r'},
    {"graphical", no_argument, 0, 'g'},
    {"iterations", required_argument, 0, 'i'},
    {"seed", required_argument, 0, 's'},
    {"temperature", required_argument, 0, 't'},
    {"output", required_argument, 0, 'o'}
};

/* Prototypes */
void        usage(char*);
void        write_grid_to_bitmap(Grid*, std::string);
void        monte_carlo(Grid*);
float       calculate_exp_ke_t(int, float);
double      calculate_time_delta(_time_point, _time_point);
void        print_progress(double, long, long, long);

/* 
 * Calculate time delta between two time points (in seconds)
 * t1 > t0 
 */
inline double calculate_time_delta(_time_point t1, _time_point t0) {
    return std::chrono::
        duration_cast<std::chrono::duration<double>>(t1 - t0).count();
}

/*
 * Fractionally divide two longs
 */
inline long double frac_long_divide(long x, long y) {
    return static_cast<long double>(x) / (static_cast<long double>(y));
}

/*
 * Calculate a random float within the specified range (inclusive)
 */
inline float rand_float_range(float low, float high) {
    return static_cast<float>(rand()) / 
        (static_cast<float>(RAND_MAX / (high - low))) + low;
}

#endif /*__SERIAL_H_*/