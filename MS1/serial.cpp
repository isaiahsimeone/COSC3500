
#include "grid.hpp"
#include "serial.hpp"

using namespace std;

// TODO: take output folder name as argument

int main(int argc, char** argv) {
    int width = 100;
    int height = 100;
    long refresh_rate = 5; // draw image every 5 iterations
    long iterations = 1000;
    float temperature = 1.5;
    unsigned int seed = 0;
    bool graphical = false;

    /* Parse arguments */
    char opt;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {

        switch (opt) {
            case 'w':
                width = stoi(optarg);
                break;
            case 'h':
                height = stoi(optarg);
                break;
            case 'r':
                refresh_rate = stol(optarg);
                break;
            case 'i':
                iterations = stol(optarg);
                break;
            case 's':
                seed = stoi(optarg);
                break;    
            case 't':
                temperature = stof(optarg);
                break;
            case 'g':
                graphical = true;
                break;
            case '?': 
                /* FALLTHROUGH */
            default:
                usage(argv[0]);
                /* NOT REACHED */
        }
    }

    if (optind < argc)
        cerr << "Ignoring " << argc - optind << " extra argument(s)" << endl;

    cout << "w: " << width << "\nh: " << height << "\nr: " << refresh_rate << "\ng: " << graphical << "\niterations: " << iterations << "\nseed: " << seed << "\ntemperature: " << temperature << endl;

    
    Grid* grid = new Grid(width, height, temperature);
    grid->randomise(seed);

    //grid->print();

    for (long i = 1; i <= iterations ;i++) {
        monte_carlo(grid);
        if (i % refresh_rate == 0) {
            cout << "\r" << static_cast<long double>(i) / ( static_cast<long double>(iterations)) * 100 << "\% complete    ";
            fflush(stdout);
            write_grid_to_bitmap(grid, ("output_img/outfile_" + to_string((int)(i/refresh_rate)) + ".bmp"));
        }
        //usleep(1000 * 1);
    }
    cout << "Done. took x seconds" << endl;

}

void monte_carlo(Grid* grid) {
    /* Randomly pick a position (i, j) */
    pair<int, int> point = grid->get_random_coords();
    int i = point.first;
    int j = point.second;

    int energy = grid->calculate_energy(i, j);
    /* If E > 0, switch the spin */
    if (energy > 0) {
        grid->switch_cell(i, j);
    /* If E < 0, pick r E [0, 1). If r < e^(2E/T), switch */
    } else if (energy < 0) {
        float r = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 0.99999));
        if (r < calculate_exp_ke_t(energy, grid->get_temperature()))
            grid->switch_cell(i, j);
    }

}

float calculate_exp_ke_t(int energy, float temperature) {
    return exp(2*energy / temperature);
}

/*
 * Write the specified grid to a bitmap image file named 'out_file_name'
 * This code has been adapted from (deusmacabre)
 * https://stackoverflow.com/questions/2654480/writing-bmp-image-in-pure-c-c-without-other-libraries
 */
void write_grid_to_bitmap(Grid* grid, string out_file_name) {
    unsigned char bmp_file_header[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
    unsigned char bmp_info_header[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};
    unsigned char bmp_padding[3] = {0, 0, 0};

    int h = grid->get_height();
    int w = grid->get_width();

    FILE* out_file = fopen(out_file_name.c_str(), "wb");
    int filesize = BITMAP_HEADER_SZ + 3 * w * h;

    unsigned char* img = (unsigned char*)malloc(3 * w * h);
    memset(img, 0, sizeof(3 * w * h));
    int colour;

    /* Create 3 byte pixels */
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            colour = (grid->get_cell(i, j) == -1 ? 255 : 0);

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


/*
 * Displays usage message to stderr
 */
void usage(char* prog_name) {
    cerr << prog_name << " pattern_file [--[w]idth grid_width] " 
            << "[--[h]eight grid_height] [--[r]efresh-rate <hz>]" 
            << " [--[g]raphical] [--[i]terations n] [--[s]eed] [--[t]emperature]" << endl;
    exit(1);
}

/*
 * Display message to stderr, then exit
 */
 void e_exit(const char* msg) {
    cerr << msg << endl;
    exit(1);
 }