
#include "grid.hpp"
#include "serial.hpp"

        Grid::Grid() {
            this->width = -1;
            this->height = -1;
            this->loaded = false;
        }


        Grid::Grid(int width = -1, int height = -1) {
            this->width = width;
            this->height = height;
            this->loaded = false;
        }

        Grid::~Grid() {
            if (!loaded)
                return ;

            for (int i = 0; i < this->height; i++)
                delete[] grid[i];
            delete[] grid;
        }

        char Grid::get_cell(int x, int y) {
            return grid[y][x];
        }

        void Grid::load_pattern_from_file(std::string input_pattern_name) {
            /* Process input grid */
            std::ifstream input(input_pattern_name);

            if (!input.is_open())
                e_exit("Failed to open input file");

            std::string line;
            int required_width = 0, required_height = 0;

            /* Calculate required width and height*/
            while (getline(input, line)) {
                /* Skip comment lines */
                if (line[0] == '!')
                    continue;
                /* Max line length */
                if ((signed)line.length() > required_width)
                    required_width = line.length();
                required_height++;
            }

            /* Warn about updated sizes */
            if (width != -1 && required_width > width)
                std::cerr << "WARN: width will be updated to fit pattern" << std::endl;
            if (height != -1 && required_height > height)
                std::cerr << "WARN: height will be updated to fit pattern" << std::endl;

            width = MAX(width, required_width);
            height = MAX(height, required_height);
            
            std::cout << "required width: " << width << " required height: " << height << std::endl;

            grid = new char*[height]();
            for (int i = 0; i < height; i++)
                grid[i] = new char[width]();
            
            /* Rewind input file */
            input.clear();
            input.seekg(0);

            /* Populate grid */
            int row = 0;
            while (std::getline(input, line)) {
                if (line[0] == '!')
                    continue;
                for (int i = 0; i < width; i++) {
                    if (i >= line.length() || line[i] == '\r' || line[i] == '\n')
                        grid[row][i] = CELL_DEAD;
                    else
                        grid[row][i] = line[i];
                }
                row++;
            }
            loaded = true;
            input.close();
        }

        void Grid::print() {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    std::cout << grid[i][j];
                }
                std::cout << std::endl;
            }
        }

        int Grid::get_width() {
            return width;
        }

        int Grid::get_height() {
            return height;
        }
