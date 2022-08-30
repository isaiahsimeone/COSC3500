#include "lattice.hpp"

/*
 * Initialise a Lattice object
 */
Lattice::Lattice(int width, int height, float temperature) {
    this->width = width;
    this->height = height;
    this->temperature = temperature;
    allocate();
}

/*
 * Deallocate the lattice object
 */
Lattice::~Lattice() {
    for (int i = 0; i < this->height; i++)
        delete[] lattice[i];
    delete[] lattice;
}

/*
 * Return the value at the cell specified by coordinates
 */
int Lattice::get_cell(int x, int y) {
    if (y >= height || y < 0)
        return 0;
    if (x >= width || x < 0)
        return 0;
    return lattice[y][x];
}

/*
 * Set the cell at the specified coordinates with
 * the specified value
 */
void Lattice::set_cell(int x, int y, int val) {
    assert(x >= 0 && x < width);
    assert(y >= 0 && y < height);

    lattice[y][x] = val;
}

/*
 * Flip the sign of the cell at the specified coordinates
 */
void Lattice::switch_cell(int x, int y) {
    assert(x >= 0 && x < width);
    assert(y >= 0 && y < height);

    lattice[y][x] *= -1;
}

/*
 * Allocate the lattice array with the specified dimensions
 */
void Lattice::allocate() {
    lattice = new int*[height]();
    for (int i = 0; i < height; i++)
        lattice[i] = new int[width]();
}

void Lattice::init_spin(int spin) {
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            lattice[i][j] = spin;
}

/*
 * Randomise lattice with +1's and -1's
 */
void Lattice::randomise(std::string seed_str = "COSC3500") {
    unsigned int seed = 0;

    /* Numerical seed provided or set seed as string hash */
    if (is_numerical(seed_str))
        seed = std::stoi(seed_str);
    else
        seed = std::hash<std::string>()(seed_str);

    /* Initialise */
    srand(seed);
    
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            lattice[i][j] = (rand() % 2 == 0 ? 1 : -1);
}

void Lattice::dump_information(FILE* f, long iteration) {

    double average_spin = 0;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (lattice[i][j] == -1) {
                average_spin++;
            } else {
                average_spin--;
            }
        }
    }
    average_spin /= width * height;

    fprintf(f, "%ld,%lf\n", iteration, average_spin);
}


int Lattice::calculate_energy_delta(int x, int y) {
    return (2 * 1/*J*/ * get_cell(x, y)) * (get_cell(x + 1, y) 
        + get_cell(x - 1, y) + get_cell(x, y + 1) + get_cell(x, y - 1));
}

/*
 * Return a pair of random lattice coordinates that are within 
 * lattice height & width
 */
std::pair<int, int> Lattice::get_random_coords() {
    int rand_x = 0 + (rand() % static_cast<int>(width));
    int rand_y = 0 + (rand() % static_cast<int>(height));
    
    return std::make_pair(rand_x, rand_y);
}

/*
 * Print the latitice to stdout replacing -1 with 'O', 
 * and +1 with a space
 */
void Lattice::print() {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << (lattice[i][j] == -1 ? '0' : ' ');
        }
        std::cout << std::endl;
    }
}

/*
 * Return the width of the lattice
 */
int Lattice::get_width() {
    return width;
}

/*
 * Return the height of the lattice
 */
int Lattice::get_height() {
    return height;
}

/*
 * Return the temperature of the lattice
 */
float Lattice::get_temperature() {
    return temperature;
}

/*
 * Write the specified lattice to a bitmap image file named 'out_file_name'
 * This code has been adapted from (deusmacabre)
 * stackoverflow.com/questions/2654480
 * /writing-bmp-image-in-pure-c-c-without-other-libraries
 */
void Lattice::write_to_bitmap(std::string out_file_name) {
    unsigned char bmp_file_header[14] 
        = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
    unsigned char bmp_info_header[40] 
        = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};
    unsigned char bmp_padding[3] = {0, 0, 0};

    int h = height;
    int w = width;

    FILE* out_file = fopen(out_file_name.c_str(), "wb");
    int filesize = BITMAP_HEADER_SZ + 3 * w * h;

    unsigned char* img = (unsigned char*)malloc(3 * w * h);
    memset(img, 0, sizeof(3 * w * h));
    int colour;

    /* Create 3 byte pixels */
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            colour = (get_cell(i, j) == -1 ? 255 : 0);

            img[(j * w + i) * 3 + 2] = (unsigned char)colour;
            img[(j * w + i) * 3 + 1] = (unsigned char)colour;
            img[(j * w + i) * 3 + 0] = (unsigned char)colour;
        }
    }

    /* Populate headers */
    bmp_file_header[2] = (unsigned char)(filesize >> 0);
    bmp_file_header[3] = (unsigned char)(filesize >> 8);
    bmp_file_header[4] = (unsigned char)(filesize >> 16);
    bmp_file_header[5] = (unsigned char)(filesize >> 24);

    bmp_info_header[4] = (unsigned char)(w >> 0);
    bmp_info_header[5] = (unsigned char)(w >> 8);
    bmp_info_header[6] = (unsigned char)(w >> 16);
    bmp_info_header[7] = (unsigned char)(w >> 24);

    bmp_info_header[8] = (unsigned char)(h >> 0);
    bmp_info_header[9] = (unsigned char)(h >> 8);
    bmp_info_header[10] = (unsigned char)(h >> 16);
    bmp_info_header[11] = (unsigned char)(h >> 24);

    /* Write headers */
    fwrite(bmp_file_header, 1, sizeof(bmp_file_header), out_file);
    fwrite(bmp_info_header, 1, sizeof(bmp_info_header), out_file);

    /* Write image pixels */
    for (int i = 0; i < h; i++) {
        fwrite(img + (w * (h - i - 1) * 3), 3, w, out_file);
        fwrite(bmp_padding, 1, (4 - (w * 3) % 4) % 4, out_file);
    }
    free(img);
    fclose(out_file);
}
