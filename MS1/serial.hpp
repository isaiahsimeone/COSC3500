#ifndef __SERIAL_H_
#define __SERIAL_H_
#include <iostream>
#include <fstream>
#include <cstring>
#include <getopt.h>
#include <unistd.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define CELL_DEAD          '.' // The cell is dead
#define CELL_DEAD_SOON     ',' // The cell will be dead on the next iteration
#define CELL_ALIVE         'O' // The cell is currently alive
#define CELL_ALIVE_SOON    'o' // The cell will be alive on the next iteration

#define BITMAP_HEADER_SZ    54

extern char* optarg;
extern int optind, opterr, optopt;

const char* const short_opts = "gw:h:r:";
const option long_opts[] = {
    {"width", required_argument, nullptr, 'w'},
    {"height", required_argument, nullptr, 'h'},
    {"refresh-rate", required_argument, nullptr, 'r'},
    {"graphical", no_argument, nullptr, 'g'}
};

enum Transition {
    T_DEAD = 0,
    T_ALIVE = 1
};

/* Prototypes */
void        usage(char*);
void        e_exit(const char*);
char**      load_pattern_from_file(std::ifstream&, int*, int*);
void        write_grid_to_bitmap(Grid*, const char*);
Transition  neighbour_check(char, char, char, char, char, char, char, char, char);
void        conway(char**, int, int);

#endif /*__SERIAL_H_*/