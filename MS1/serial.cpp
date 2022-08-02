
#include "grid.hpp"
#include "serial.hpp"

using namespace std;

int main(int argc, char** argv) {
    int width = 100;
    int height = 100;
    long draw_rate = 5; // draw image every 5 iterations
    long iterations = 1000;
    float temperature = 1.5;
    string seed = "";
    bool graphical = false;
    string output_dir_name = "output_img/";

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
                draw_rate = stol(optarg);
                break;
            case 'i':
                iterations = stol(optarg);
                break;
            case 's':
                seed = optarg;
                break;    
            case 't':
                temperature = stof(optarg);
                break;
            case 'g':
                graphical = true;
                break;
            case 'o':
                output_dir_name = optarg;
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

    cout << "Width:       " << width << "px\n"
         << "Height:      " << height << "px\n"
         << "Refresh:     " << draw_rate << " iterations between each image\n"
         << "Iterations:  " << iterations << "\n"
         << "Seed:        " << seed << "\n"
         << "Temperature: " << temperature << "\n"
         << "Output dir:  " << output_dir_name << "\n";

    /* Calculate how much space this will take */
    if (graphical) {
        long required_size = (BITMAP_HEADER_SZ + 3 * width * height) * (iterations / draw_rate);
        cout << "This simulation will consume " 
            << required_size / BYTES_PER_MEGABYTE << "MB of disk space and create "
            << iterations/draw_rate << " images with the specified parameters" << endl;
    }
    cout << endl;

    Grid* grid = new Grid(width, height, temperature);
    grid->randomise(seed);

    //grid->print();
    cout << "Initialised\n" << endl;

    /* Create output folder if it doesn't exist */
    if (graphical) {
        mkdir(output_dir_name.c_str(), 0777);
    }

    /* Write first image */
    if (graphical)
        write_grid_to_bitmap(grid, (output_dir_name + "/outfile_0.bmp"));

    /* Initialise timer */
    _time_point start, end;
    _time_point t0 = _clock::now();
    start = t0;
    for (long i = 1; i <= iterations; i++) {
        monte_carlo(grid);
        if (i % draw_rate == 0) {
            
            end = _clock::now();
            print_progress(calculate_time_delta(end, start), i, iterations, draw_rate);
            start = _clock::now();

            if (graphical)
                write_grid_to_bitmap(grid, (output_dir_name + "/outfile_" + to_string((int)(i/draw_rate)) + ".bmp"));
        }
    }

    _time_point t1 = _clock::now();

    auto printable_start = t0.time_since_epoch().count();
    auto printable_end = t1.time_since_epoch().count();

    cout << "\n\nStart time: " << printable_start
         << "\nEnd time:   " << printable_end
         << "\nDelta:      " << printable_end - printable_start << endl;

    /* This has overflow potential */
    cout << "\n[*] Done. Took " << calculate_time_delta(t1, t0) << " seconds\n" << endl;
}

void print_progress(double time_between_draws, long i, long iterations, long draw_rate) {
    /* How long did it take from last draw to this draw */
    long double iterations_remaining = iterations - i;
    /* Calculate remaining time */
    int remaining_time = ceil((iterations_remaining / draw_rate) * time_between_draws * 100) / 100;
    int h = remaining_time / 3600;
    int m = (remaining_time % 3600) / 60;
    int s = (remaining_time % 3600) % 60;
    string remaining = to_string(h) + "h" + to_string(m) + "m" + to_string(s) + "s";

    cout << "\r" << frac_long_divide(i, iterations) * 100 << "% complete. ~"
         << remaining << " remaining       ";

    fflush(stdout);
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
    cerr << prog_name << " pattern_file [--[w]idth] " 
            << "[--[h]eight] [--[r]efresh-rate]" 
            << " [--[g]raphical] [--[i]terations] [--[s]eed] [--[t]emperature]"
            << " [--[o]utput <output_dir_name>]" << endl;
    exit(1);
}

/*
 * Display message to stderr, then exit
 */
 void e_exit(const char* msg) {
    cerr << msg << endl;
    exit(1);
 }