#ifndef __GRID_H_
#define __GRID_H_

#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>

class Grid {
    public:
        std::pair<int,int>  get_random_coords();
        char                get_cell(int, int);
        void                randomise(unsigned int);
        int                 get_width();
        int                 get_height();
        void                print();
        int                 calculate_energy(int, int);
                            Grid(int, int);
                            ~Grid();
    private:
        void                allocate();
        int                 width;
        int                 height;
        char**              grid;
        bool                allocated;
};

#endif /*__GRID_H_*/