#ifndef __LATTICE_H_
#define __LATTICE_H_

#include <fstream>
#include <iostream>
#include <string>
#include <cassert>  /* assert */
#include <cstring>  /* memset */

#include "util.hpp"

class Lattice {
    public:
        int*                lattice;
        int                 dimension;
        std::pair<int,int>  get_random_coords();
        unsigned long long  get_seed();
        int                 get_cell(int, int);
        void                randomise(std::string);
        int                 get_dimension();
        int                 calculate_energy(int, int);
        void                switch_cell(int, int);
        float               get_temperature();
        void                write_to_bitmap(std::string);
        void                dump_information(FILE*, long);
                            Lattice(int, float);
                            ~Lattice();
    private:
        void                allocate();
        unsigned long long  seed;
        float               temperature;
};

#endif /*__LATTICE_H_*/
