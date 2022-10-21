#include "lattice.hpp"

/*
 * Initialise a Lattice object
 */
Lattice::Lattice(int dimension, float temperature) {
    this->lattice = nullptr;
    this->dimension = dimension;
    this->temperature = temperature;
    this->seed = 3500;
    allocate();
}

/*
 * Deallocate the lattice object
 */
Lattice::~Lattice() {
    delete[] lattice;
}

/*
 * Return the value at the cell specified by coordinates
 */
int Lattice::get_cell(int x, int y) const {
    if (y >= dimension || y < 0)
        return 0;
    if (x >= dimension || x < 0)
        return 0;
    return lattice[x * dimension + y];
}

/*
 * Allocate the lattice array with the specified dimensions
 */
void Lattice::allocate() {
    lattice = new int[dimension * dimension]();
}

/*
 * Randomise lattice with +1's and -1's
 */
void Lattice::randomise(const std::string& seed_str = "COSC3500") {
    unsigned int numerical_seed;

    /* Numerical seed provided or set seed as string hash */
    if (is_numerical(seed_str))
        numerical_seed = std::stoi(seed_str);
    else
        numerical_seed = std::hash<std::string>()(seed_str);

    /* Initialise */
    srand(numerical_seed);
    this->seed = numerical_seed;
    
    for (int i = 0; i < dimension * dimension; i++)
        lattice[i] = (rand() % 2 == 0 ? 1 : -1);
}

/*
 * Output information about the lattice to the specified
 * file.
 */
void Lattice::dump_information(FILE* f, long iteration) const {
    double average_spin = 0;

    for (int i = 0; i < dimension * dimension; i++) {
        if (lattice[i] == -1)
            average_spin++;
        else
            average_spin--;
    }

    average_spin /= dimension * dimension;

    fprintf(f, "%ld,%lf\n", iteration, average_spin);
}

/*
 * Return the width of the lattice
 */
int Lattice::get_dimension() const {
    return dimension;
}

/*
 * Return the temperature that the lattice was initialised with
 */
float Lattice::get_temperature() const {
    return temperature;
}


/*
 * Return the seed that was used to initialise the lattice
 * in numerical form (string seeds are lost)
 */
unsigned long long Lattice::get_seed() const {
    return seed;
}

/*
 * Write the specified lattice to a bitmap image file named 'out_file_name'
 * This code has been adapted from (deusmacabre)
 * stackoverflow.com/questions/2654480
 * /writing-bmp-image-in-pure-c-c-without-other-libraries
 */
void Lattice::write_to_bitmap(const std::string& out_file_name) const {
    unsigned char bmp_file_header[14] 
        = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
    unsigned char bmp_info_header[40] 
        = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};
    unsigned char bmp_padding[3] = {0, 0, 0};

    int h = dimension;
    int w = dimension;

    FILE* out_file = fopen(out_file_name.c_str(), "wb");
    int filesize = BITMAP_HEADER_SZ + 3 * w * h;

    auto* img = (unsigned char*)malloc(3 * w * h);
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
