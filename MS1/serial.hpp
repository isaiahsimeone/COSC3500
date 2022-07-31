#ifndef __SERIAL_H_
#define __SERIAL_H_
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <getopt.h>
#include <unistd.h>
#include <assert.h>
#include <sys/stat.h>
#include <fcntl.h>

#define BITMAP_HEADER_SZ    54
#define BYTES_PER_MEGABYTE  1000000

typedef std::chrono::time_point<std::chrono::high_resolution_clock> _time_point;
typedef std::chrono::high_resolution_clock                          _clock;

extern char* optarg;
extern int optind, opterr, optopt;

const char* const short_opts = "gw:h:r:i:s:t:o:";
const option long_opts[] = {
    {"width", required_argument, nullptr, 'w'},
    {"height", required_argument, nullptr, 'h'},
    {"refresh-rate", required_argument, nullptr, 'r'},
    {"graphical", no_argument, nullptr, 'g'},
    {"iterations", required_argument, nullptr, 'i'},
    {"seed", required_argument, nullptr, 's'},
    {"temperature", required_argument, nullptr, 't'},
    {"output", required_argument, nullptr, 'o'}
};

/* Prototypes */
void        usage(char*);
void        e_exit(const char*);
void        write_grid_to_bitmap(Grid*, std::string);
void        monte_carlo(Grid*);
float       calculate_exp_ke_t(int, float);
double      calculate_time_delta(_time_point, _time_point);
void        print_progress(double, long, long, long);

/* t1 > t0 */
inline double calculate_time_delta(_time_point t1, _time_point t0) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
}

inline long double frac_long_divide(long x, long y) {
    return static_cast<long double>(x) / (static_cast<long double>(y));
}

#endif /*__SERIAL_H_*/