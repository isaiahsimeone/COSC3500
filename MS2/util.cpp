#include "util.hpp"

/* 
 * Calculate time delta between two time points (in seconds)
 * t1 > t0 
 */
double calculate_time_delta(_time_point t1, _time_point t0) {
    return std::chrono::
        duration_cast<std::chrono::duration<double>>(t1 - t0).count();
}

/*
 * Fractionally divide two longs
 */
long double frac_long_divide(long x, long y) {
    return static_cast<long double>(x) / (static_cast<long double>(y));
}

/*
 * Determines whether the specified string, s, is composed
 * of only digits.
 */
bool is_numerical(const std::string& s) {
    for (auto c : s)
        if (!std::isdigit(c))
            return false;
    return true;
}