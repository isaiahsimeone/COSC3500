#include "grid.hpp"
#include "serial.hpp"

Grid::Grid(int width, int height, float temperature) {
    this->width = width;
    this->height = height;
    this->temperature = temperature;
    this->allocated = false;
}

Grid::~Grid() {
    if (!allocated)
        return ;
    
    for (int i = 0; i < this->height; i++)
        delete[] grid[i];
    delete[] grid;
}

int Grid::get_cell(int x, int y) {
    if (y >= height || y < 0)
        return 0;
    if (x >= width || x < 0)
        return 0;
    return grid[y][x];
}

void Grid::set_cell(int x, int y, int val) {
    assert(x >= 0 && x < width);
    assert(y >= 0 && y < height);

    grid[y][x] = val;
}

void Grid::switch_cell(int x, int y) {
    assert(x >= 0 && x < width);
    assert(y >= 0 && y < height);

    grid[y][x] *= -1;
}

void Grid::allocate() {
    grid = new int*[height]();
    for (int i = 0; i < height; i++)
        grid[i] = new int[width]();
    allocated = true;
}

/*
 * Randomise grid with +1's and -1's
 */
void Grid::randomise(std::string seed_str = "") {
    unsigned int seed = 0;

    /* Numerical seed provided or set seed as string hash */
    if (is_numerical(seed_str))
        seed = std::stoi(seed_str);
    else
        seed = std::hash<std::string>()(seed_str);

    /* Initialise */
    srand(seed);

    if (!allocated)
        allocate();
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            grid[i][j] = (rand() % 2 == 0 ? 1 : -1);
        }
    }
}

int Grid::calculate_energy(int x, int y) {
    return (-1 * get_cell(x, y)) * (get_cell(x + 1, y) + get_cell(x - 1, y) + get_cell(x, y + 1) + get_cell(x, y - 1));
}

/*
 * Return a pair of random grid coordinates that are within grid height & width 
 */
std::pair<int, int> Grid::get_random_coords() {
    int rand_x = 0 + (rand() % static_cast<int>(width + 1));
    int rand_y = 0 + (rand() % static_cast<int>(height + 1));
    
    return std::make_pair(rand_x, rand_y);
}

void Grid::print() {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << (grid[i][j] == -1 ? '0' : ' ');
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

float Grid::get_temperature() {
    return temperature;
}