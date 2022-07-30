#ifndef __SERIAL_H_
#define __SERIAL_H_
#include <iostream>
#include <fstream>
#include <cstring>
#include <getopt.h>
#include <unistd.h>

#define BITMAP_HEADER_SZ    54

extern char* optarg;
extern int optind, opterr, optopt;

const char* const short_opts = "gw:h:r:i:s:";
const option long_opts[] = {
    {"width", required_argument, nullptr, 'w'},
    {"height", required_argument, nullptr, 'h'},
    {"refresh-rate", required_argument, nullptr, 'r'},
    {"graphical", no_argument, nullptr, 'g'},
    {"iterations", required_argument, nullptr, 'i'},
    {"seed", required_argument, nullptr, 's'}
};

/* Prototypes */
void        usage(char*);
void        e_exit(const char*);
void        write_grid_to_bitmap(Grid*, std::string);
void        monte_carlo(Grid*);

#endif /*__SERIAL_H_*/