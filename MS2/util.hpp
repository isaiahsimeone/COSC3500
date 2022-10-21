#ifndef __UTIL_H_
#define __UTIL_H_

#include <chrono>
#include <string>
#include <iostream>

#define BITMAP_HEADER_SZ    54
#define BYTES_PER_MEGABYTE  1000000

#define COLOUR_GREEN        "\x1B[32m"
#define COLOUR_RESET        "\x1B[0m"

typedef std::chrono::time_point<std::chrono::high_resolution_clock> _time_point;
typedef std::chrono::high_resolution_clock                          _clock;

/* Prototypes */
double      calculate_time_delta(_time_point, _time_point);
long double frac_long_divide(long, long);
bool        is_numerical(const std::string&);

#endif /* __UTIL_H_ */
