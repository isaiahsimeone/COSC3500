#include "parallel.hpp"

__global__ void        evaluate_disjoint_component(int*, float*, int, float, int);

using namespace std;

/*
 * Monte carlo ising. Simple usage: ./parallel --help
 * Basic example: ./parallel -w 250 -h 250 -i 100000000
 * With graphics: ./parallel -w 250 -h 250 -i 100000000 -r 100000 -g -o output_img
 */
int main(int argc, char** argv) {
    /* Dimension of the lattice */
    int dimension = 100;
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
    /* The number of lattice division to perform. Each lattice division is
     * managed in parallel, independently from other divisions.
     * one division divides the lattice into four pieces, a second division into 16, 
     * a third into 64, (4**n) */
    int lattice_divisions = 0;
    /* Initial configuration of the lattice based on this seed */
    string seed = "COSC3500";
    /* The location to write images of the lattice */
    string output_dir_name = "output_img/";
    FILE* info_dump_file = NULL;

    /* Parse arguments */
    char opt;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, 0)) != -1) {
        switch (opt) {
            case 'n':
                dimension = stoi(optarg);
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
            case 'l':
                lattice_divisions = stoi(optarg);
                break;
            case 'c':
                thread_count = stoi(optarg);
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
    cout << "Dimension:       " << dimension << "px\n"
         << "Refresh:     " << update_rate << " iterations between each image\n"
         << "Iterations:  " << iterations << "\n"
         << "Divisions:   " << lattice_divisions << "\n"
         << "Threads:     " << thread_count << "\n"
         << "Seed:        " << seed << "\n"
         << "Temperature: " << temperature << "\n"
         << "Output dir:  " << output_dir_name << "\n";

    //if (dimension % 2 != 0)
    //    msg_err_exit("Lattice dimension must be even");


    /* Calculate how much disk space this simulation will take */
    if (graphical) {
        long required_size = (BITMAP_HEADER_SZ + 3 * dimension * dimension) 
            * (iterations / update_rate);
        cout << "This simulation will consume "
            << required_size / BYTES_PER_MEGABYTE 
            << "MB of disk space and create "
            << iterations / update_rate 
            << " images with the specified parameters" << endl;
    }
    cout << endl;

    /* Randomise lattice with specified seed */
    Lattice* lattice = new Lattice(dimension, temperature);
    lattice->randomise(seed);

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

    cout << "Allocating VRAM..." << endl;

    size_t lattice_sz = lattice->get_dimension() * lattice->get_dimension();
    int block_count = floor(lattice_sz / thread_count) + 1;
    cout << "Block count is " << block_count << endl;


    /* Initialise timer */
    _time_point start, end;
    _time_point t0 = _clock::now();
    start = t0;

    float* random_device, *random_host;
    int* lattice_device;
    random_host = (float*)malloc(sizeof(float) * lattice_sz);

    // Cuda alloc
    CUDACheckErr(cudaMalloc(&lattice_device, lattice_sz * sizeof(int)));
    CUDACheckErr(cudaMalloc(&random_device, lattice_sz * sizeof(float)));

    // Copy lattice to device
    CUDACheckErr(cudaMemcpy(lattice_device, lattice->lattice, lattice_sz * sizeof(float), cudaMemcpyHostToDevice));


    for (long i = 1; i <= iterations; i++) {
        // Populate random floats
        for (int j = 0; j < lattice_sz; j++)
            random_host[j] = rand_float_range(0, 0.9999);
        // Copy those random floats to the device
        CUDACheckErr(cudaMemcpy(random_device, random_host, lattice_sz * sizeof(float), cudaMemcpyHostToDevice));
        
        // Do lattice sites with even index
        evaluate_disjoint_component<<<block_count, thread_count>>>(lattice_device, random_device, dimension, temperature, PARITY_EVEN);
        CUDACheckErr(cudaDeviceSynchronize());
        // Do lattice sites with odd index
        evaluate_disjoint_component<<<block_count, thread_count>>>(lattice_device, random_device, dimension, temperature, PARITY_ODD);
        CUDACheckErr(cudaDeviceSynchronize());
        
        if (i % update_rate == 0) {
            CUDACheckErr(cudaMemcpy(lattice->lattice, lattice_device, lattice_sz * sizeof(int), cudaMemcpyDeviceToHost));

            if (graphical)
                lattice->write_to_bitmap((output_dir_name + "/outfile_" + to_string((int)(i/update_rate)) + ".bmp"));
        }
    }

    CUDACheckErr(cudaDeviceSynchronize());
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


__global__
void evaluate_disjoint_component(int* lattice, float* random, int lattice_dimension, float temperature, int parity) {
    // This is the index of a specific lattice site. A cuda thread is
    // run for every lattice site
    int site_index = blockDim.x  * blockIdx.x + threadIdx.x;

    // There will be left over threads. But we don't want only want enough for each lattice site
    if (site_index >= lattice_dimension * lattice_dimension)
        return ;

    // The parity argument determines which configuration of lattice sites we're evaluating.
    // There are two disjoint configurations which are handled independently. 
    if (lattice_dimension % 2 != 0 && site_index % 2 == parity) // false and false
        return ;

    // IF parity odd (start 1)
    // IF row is even and site_index is odd, break
    // IF row is odd  and site_index is even, break
    int row = site_index / lattice_dimension;
    
    if (lattice_dimension % 2 == 0) { //true
        if (parity == PARITY_ODD) { //true
            if (row % 2 == 0 && site_index % 2 != 0) { //true && odd
                return ;
            } else if (row % 2 != 0 && site_index % 2 == 0) {
                return ;
            }
        }

        // IF parity even (start 0)
        // IF row is even and site_index is even, break
        // IF row is odd and site_index is odd, break

        if (parity == PARITY_EVEN) {
            if (row % 2 == 0 && site_index % 2 == 0) {
                return ;
            } else if (row % 2 != 0 && site_index % 2 != 0) {
                return ;
            }
        }
    }

    int north_neighour = (site_index >= lattice_dimension ? site_index - lattice_dimension : -1);
    int east_neighbour = (site_index + 1 < lattice_dimension * lattice_dimension ? site_index + 1 : -1);
    int south_neighbour= (site_index < lattice_dimension * (lattice_dimension - 1) ? site_index + lattice_dimension : -1);
    int west_neighbour = (site_index - 1 >= 0 ? site_index - 1 : -1);

    int adjacent_sum = 0;

    if (north_neighour != -1)
        adjacent_sum += lattice[north_neighour];
    if (east_neighbour != -1)
        adjacent_sum += lattice[east_neighbour];
    if (south_neighbour != -1)
        adjacent_sum += lattice[south_neighbour];
    if (west_neighbour != -1)
        adjacent_sum += lattice[west_neighbour];

    int energy = 2 * adjacent_sum * lattice[site_index];


    if (energy < 0 || random[site_index] < exp(-1 * energy / temperature))
        lattice[site_index] *= -1;


    printf("I calculate %d which has NESW: %d %d %d %d\n", site_index, north_neighour, east_neighbour, south_neighbour, west_neighbour);
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

