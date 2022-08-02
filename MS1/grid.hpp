#ifndef __GRID_H_
#define __GRID_H_

#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>

inline bool is_numerical(std::string s) {
    for (auto c : s)
        if (!std::isdigit(c))
            return false;
    return true;
}

class Grid {
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
                            Grid(int, int, float);
                            ~Grid();
    private:
        void                allocate();
        int                 width;
        int                 height;
        int**               grid;
        float               temperature;
        bool                allocated;
};

#endif /*__GRID_H_*/