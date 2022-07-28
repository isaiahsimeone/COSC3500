
#include "grid.hpp"
#include "serial.hpp"

using namespace std;

// TODO: take output folder name as argument

int main(int argc, char** argv) {
    int width = -1;
    int height = -1;
    int refresh_rate = 5; // hz
    bool graphical = false;
    string input_pattern_name = "";

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
                refresh_rate = stoi(optarg);
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

    /* Input file name */
    if (optind >= argc)
        usage(argv[0]);
    else
        input_pattern_name = argv[optind++];
        
    if (optind < argc)
        cerr << "Ignoring " << argc - optind << " extra argument(s)" << endl;

    cout << "w: " << width << "\nh: " << height << "\nr: " << refresh_rate << "\nf: " << input_pattern_name << "\ng: " << graphical << endl;


    
    Grid* grid = new Grid();
    grid->load_pattern_from_file(input_pattern_name);

    grid->print();

    for (int i = 0; ;i++) {
        //conway(grid, width, height);
        write_grid_to_bitmap(grid, ("output_img/outfile_" + to_string(i) + ".bmp").c_str());
        usleep(1000 * 100);
    }

}

void conway(char** grid, int width, int height) {
   
}

/*
 * Determine's what should happen to the specified cell given 
 * information of it's north, east, south and west neighbours
 * according to the rules of conway's game of life
 */
Transition neighbour_check(char self, char N, char NE, char E, char SE,
        char S, char SW, char W, char NW) {
    int alive_neighbours = (N == CELL_ALIVE) + (E == CELL_ALIVE) +
        (S == CELL_ALIVE) + (W == CELL_ALIVE) + (NE == CELL_ALIVE) +
        (SE == CELL_ALIVE) + (SW == CELL_ALIVE) + (NW == CELL_ALIVE);

    /* Any dead cell with exactly three live neighbours becomes a live cell */
    if (self == CELL_DEAD && alive_neighbours == 3)
        return T_ALIVE;
    /* Any live cell with fewer than two live neighbours dies */
    if (alive_neighbours < 2)
        return T_DEAD;
    /* Any live cell with two or three live neighbours lives on */
    if (alive_neighbours == 2 || alive_neighbours == 3)
        return T_ALIVE;
    /* Any live cell with more than three live neighbours dies */
    if (alive_neighbours > 3)
        return T_DEAD;
    
    return T_DEAD; /* NOTREACHED */
}

/*
 * Write the specified grid to a bitmap image file named 'out_file_name'
 * This code has been adapted from (deusmacabre)
 * https://stackoverflow.com/questions/2654480/writing-bmp-image-in-pure-c-c-without-other-libraries
 */
void write_grid_to_bitmap(Grid* grid, const char* out_file_name) {
    unsigned char bmp_file_header[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
    unsigned char bmp_info_header[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};
    unsigned char bmp_padding[3] = {0,0,0};

    int h = grid->get_height();
    int w = grid->get_width();

    cout << h << " " << w << endl;

    FILE* out_file = fopen(out_file_name, "wb");
    int filesize = BITMAP_HEADER_SZ + 3 * w * h;

    unsigned char* img = (unsigned char*)malloc(3 * w * h);
    memset(img, 0, sizeof(3 * w * h));
    int colour;

    /* Create 3 byte pixels */
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            colour = (grid->get_cell(i, j) == CELL_DEAD ? 0 : 255);

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
    fclose(out_file);
}


/*
 * Displays usage message to stderr
 */
void usage(char* prog_name) {
    cerr << prog_name << " pattern_file [--[w]idth grid_width] " 
            << "[--[h]eight grid_height] [--[r]efresh-rate <hz>]" 
            << " [--[g]raphical]" << endl;
    exit(1);
}

/*
 * Display message to stderr, then exit
 */
 void e_exit(const char* msg) {
    cerr << msg << endl;
    exit(1);
 }