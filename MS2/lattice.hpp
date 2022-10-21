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
        unsigned long long  get_seed() const;
        int                 get_cell(int, int) const;
        void                randomise(const std::string&);
        int                 get_dimension() const;
        float               get_temperature() const;
        void                write_to_bitmap(const std::string&) const;
        void                dump_information(FILE*, long) const;
                            Lattice(int, float);
                            ~Lattice();
    private:
        void                allocate();
        unsigned long long  seed;
        float               temperature;
};

#endif /*__LATTICE_H_*/
