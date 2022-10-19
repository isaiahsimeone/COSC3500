#ifndef __UTIL_H_
#define __UTIL_H_

#include <chrono>
#include <string>
#include <iostream>

#define BITMAP_HEADER_SZ    54
#define BYTES_PER_MEGABYTE  1000000

#define COLOUR_GREEN        "\x1B[32m"
#define COLOUR_AMBER        "\x1B[33m"
#define COLOUR_RED          "\x1B[31m"
#define COLOUR_RESET        "\x1B[0m"

typedef std::chrono::time_point<std::chrono::high_resolution_clock> _time_point;
typedef std::chrono::high_resolution_clock                          _clock;

/* Prototypes */
double      calculate_time_delta(_time_point, _time_point);
long double frac_long_divide(long, long);
float       rand_float_range(float, float);
bool        is_numerical(std::string);
double      calculate_time_delta(_time_point, _time_point);
bool        is_power_of_two(int);
void        msg_warn(std::string);
void        msg_err_exit(std::string);

#endif /* __UTIL_H_ */
