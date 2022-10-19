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
 * Calculate a random float within the specified range (inclusive)
 */
float rand_float_range(float low, float high) {
    return static_cast<float>(rand()) / 
        (static_cast<float>(RAND_MAX / (high - low))) + low;
}

/*
 * Determines whether the specified string, s, is comprised
 * of only digits.
 */
bool is_numerical(std::string s) {
    for (auto c : s)
        if (!std::isdigit(c))
            return false;
    return true;
}

/*
 * Determine if the specified number, n, is a power of two
 */
bool is_power_of_two(int n) {
    return (n != 0) && (n & (n-1)) == 0;
}

/*
 * Write an amber coloured warning message to stderr
 */
void msg_warn(std::string msg) {
    std::cerr << COLOUR_AMBER << msg << COLOUR_RESET << std::endl;
}  

/*
 * Write a red coloured error message to stderr and abort the program
 */
void msg_err_exit(std::string msg) {
    std::cerr << COLOUR_RED << msg << COLOUR_RESET << std::endl;
    abort();
}