#ifndef __GRID_H_
#define __GRID_H_

#include <fstream>
#include <iostream>
#include <string>

class Grid {
    public:
        char    get_cell(int, int);
        void    load_pattern_from_file(std::string);
        int     get_width();
        int     get_height();
        void    print();                
                Grid();
                Grid(int, int);
                ~Grid();
    private:
        int     width;
        int     height;
        char**  grid;
        bool    loaded;
};

#endif /*__GRID_H_*/