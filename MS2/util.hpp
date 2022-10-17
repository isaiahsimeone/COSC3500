#ifndef __UTIL_H_
#define __UTIL_H_

#include <chrono>
#include <string>

#define BITMAP_HEADER_SZ    54
#define BYTES_PER_MEGABYTE  1000000

typedef std::chrono::time_point<std::chrono::high_resolution_clock> _time_point;
typedef std::chrono::high_resolution_clock                          _clock;

/* Prototypes */
double      calculate_time_delta(_time_point, _time_point);
long double frac_long_divide(long, long);
float       rand_float_range(float, float);
bool        is_numerical(std::string);
double      calculate_time_delta(_time_point, _time_point);
bool        is_power_of_two(int);

#endif /* __UTIL_H_ */
