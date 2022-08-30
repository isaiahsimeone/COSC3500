#include "serial.hpp"

using namespace std;

/*
 * Monte carlo ising. Simple usage: ./serial --help
 */
int main(int argc, char** argv) {
    int width = 100;
    int height = 100;
    long draw_rate = 0;
    long iterations = 1000;
    float temperature = 1.5;
    bool graphical = false;
    bool dump_information = false;
    string seed = "COSC3500";
    string output_dir_name = "output_img/";
    FILE* info_dump_file = NULL;

    /* Parse arguments */
    char opt;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, 0)) != -1) {
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
            case 'd':
                dump_information = true;
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

    /* Too many arguments specified */
    if (optind < argc)
        cerr << "Ignoring " << argc - optind << " extra argument(s)" << endl;

    /* If unspecified, print progress every 1% */
    if (draw_rate == 0)
        draw_rate = iterations * 0.01;

    /* Display selected settings */
    cout << "Width:       " << width << "px\n"
         << "Height:      " << height << "px\n"
         << "Refresh:     " << draw_rate << " iterations between each image\n"
         << "Iterations:  " << iterations << "\n"
         << "Seed:        " << seed << "\n"
         << "Temperature: " << temperature << "\n"
         << "Output dir:  " << output_dir_name << "\n";

    /* Calculate how much disk space this simulation will take */
    if (graphical) {
        long required_size = (BITMAP_HEADER_SZ + 3 * width * height) 
            * (iterations / draw_rate);
        cout << "This simulation will consume " 
            << required_size / BYTES_PER_MEGABYTE 
            << "MB of disk space and create "
            << iterations/draw_rate 
            << " images with the specified parameters" << endl;
    }
    cout << endl;

    /* Randomise lattice with specified seed */
    Lattice* lattice = new Lattice(width, height, temperature);
    lattice->randomise(seed);
    
    cout << "Initialised\n" << endl;

    /* Create output folder if it doesn't exist */
    if (graphical) {
        mkdir(output_dir_name.c_str(), 0777);
        /* Write first image */
        lattice->write_to_bitmap(output_dir_name + "/outfile_0.bmp");
    }

    if (dump_information)
        info_dump_file = fopen(("information_" + to_string(iterations) + "_" + to_string(temperature) + ".csv").c_str(), "w");

    /* Initialise timer */
    _time_point start, end;
    _time_point t0 = _clock::now();
    start = t0;

    /* Monte carlo loop */
    for (long i = 1; i <= iterations; i++) {
        metropolis(lattice);
        if (i % draw_rate == 0) {
            
            end = _clock::now();
            print_progress(calculate_time_delta(end, start), i, iterations,
                draw_rate);
            start = _clock::now();

            if (graphical)
                lattice->write_to_bitmap((output_dir_name + "/outfile_" 
                    + to_string((int)(i/draw_rate)) + ".bmp"));
            if (dump_information)
                lattice->dump_information(info_dump_file, i);
        }
    }

    _time_point t1 = _clock::now();

    if (dump_information)
        fclose(info_dump_file);

    auto printable_start = t0.time_since_epoch().count();
    auto printable_end = t1.time_since_epoch().count();

    cout << "\n\nStart time: " << printable_start
         << "\nEnd time:   " << printable_end
         << "\nDelta:      " << printable_end - printable_start << endl;

    cout << COLOUR_GREEN << "\n[*] Done. Took " << calculate_time_delta(t1, t0) 
         << " seconds\n" << COLOUR_RESET << endl;
}

/*
 * Display the progress of the simulation to stdout
 */
void print_progress(double time_between_draws, long i, long iterations,
        long draw_rate) {
    /* How long did it take from last draw to this draw */
    long double iterations_remaining = iterations - i;

    /* Calculate remaining time */
    int remaining_time = ceil((iterations_remaining / draw_rate) 
        * time_between_draws * 100) / 100;
    int h = remaining_time / 3600;
    int m = (remaining_time % 3600) / 60;
    int s = (remaining_time % 3600) % 60;
    string remaining = to_string(h) + "h" + to_string(m) + "m" 
        + to_string(s) + "s";

    /* x% complete ~HHMMSS remaining */
    cout << "\r" << frac_long_divide(i, iterations) * 100 
         << "% complete. ~"
         << remaining << " remaining       ";

    fflush(stdout);
}

/*
 * Perform an iteration of the Monte Carlo Metropolis
 * algorithm on the specified lattice
 */
void metropolis(Lattice* lattice) {
    /* Randomly pick a position (i, j) */
    pair<int, int> point = lattice->get_random_coords();
    int i = point.first;
    int j = point.second;

    int energy_change = lattice->calculate_energy_delta(i, j);
    
    if (energy_change < 0 || rand_float_range(0,0.999999) < exp(-1 * energy_change / (lattice->get_temperature()))) {
        lattice->switch_cell(i, j);
    }
}

float calculate_exp_ke_t(int energy, float temperature) {
    return exp(energy * 2 / temperature);
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
