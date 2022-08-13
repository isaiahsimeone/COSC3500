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
        std::pair<int,int>  get_random_coords();
        int                 get_cell(int, int);
        void                randomise(std::string);
        int                 get_width();
        int                 get_height();
        void                print();
        int                 calculate_energy(int, int);
        void                set_cell(int, int, int);
        void                switch_cell(int, int);
        float               get_temperature();
        void                write_to_bitmap(std::string);
                            Lattice(int, int, float);
                            ~Lattice();
    private:
        void                allocate();
        int                 width;
        int                 height;
        int**               lattice;
        float               temperature;
};

#endif /*__LATTICE_H_*/
