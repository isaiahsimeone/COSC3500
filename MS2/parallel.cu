#include "parallel.hpp"

using namespace std;

/*
 * Monte carlo ising. Simple usage: ./parallel --help
 * Basic example: ./parallel -w 250 -h 250 -i 100000000
 * With graphics: ./parallel -w 250 -h 250 -i 100000000 -r 100000 -g -o output_img
 */
int main(int argc, char** argv) {
    /* Width of the lattice */
    int width = 100;
    /* Height of the lattice */
    int height = 100;
    /* Number of iterations */
    long iterations = 1000;
    /* The temperature of the lattice */
    float temperature = 1.5;
    /* Should the program produce images of the lattice? */
    bool graphical = false;
    /* frequency that image of the lattice is captured (default every 1% of progress) */
    long update_rate = iterations * 0.01;
    /* Should the program write lattice statistics (e.g. average spin) ? */
    bool dump_information = false;
    /* Number of CUDA threads to use */
    int thread_count = 128;
    /* Initial configuration of the lattice based on this seed */
    string seed = "COSC3500";
    /* The location to write images of the lattice */
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
                update_rate = stol(optarg);
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

    /* Display selected settings */
    cout << "Width:       " << width << "px\n"
         << "Height:      " << height << "px\n"
         << "Refresh:     " << update_rate << " iterations between each image\n"
         << "Iterations:  " << iterations << "\n"
         << "Seed:        " << seed << "\n"
         << "Temperature: " << temperature << "\n"
         << "Output dir:  " << output_dir_name << "\n";

    /* Calculate how much disk space this simulation will take */
    if (graphical) {
        long required_size = (BITMAP_HEADER_SZ + 3 * width * height) 
            * (iterations / update_rate);
        cout << "This simulation will consume " 
            << required_size / BYTES_PER_MEGABYTE 
            << "MB of disk space and create "
            << iterations / update_rate 
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

    /* Create a CSV file to record lattice information to */
    if (dump_information)
        info_dump_file = fopen(("information_" + to_string(iterations) + "_" 
            + to_string(temperature) + ".csv").c_str(), "w");



    // Implement the parallel metropolis algorithm here.
    // I am pretty sure that because positive and negatively charged spins
    // don't interact, they can be treated as disjoint. Which means
    // That potentially, we can consider all spins of a certain charge
    // denomination at once


    /* Initialise timer */
    _time_point start, end;
    _time_point t0 = _clock::now();
    start = t0;

    /* Parallel Monte carlo loop here */
    // We can update all black squares at once, and all white squares at once in a loop

    //for (int i = 0; i < real_iterations; i++) {
        /* Update first checkboard configuration */

        /* Update second, disjoint checkerboard configuration */
    //}



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
 * TODO: update 
 * Displays usage message to stderr 
 */
void usage(char* prog_name) {
    cerr << prog_name << " pattern_file [--[w]idth] " 
            << "[--[h]eight] [--[r]efresh-rate]" 
            << " [--[g]raphical] [--[i]terations] [--[s]eed] [--[t]emperature]"
            << " [--[o]utput <output_dir_name>]" << endl;
    exit(1);
}
