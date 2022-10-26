#include "parallel.hpp"


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
    /* frequency that image of the lattice is captured */
    long update_rate = 100;
    /* Should the program write lattice statistics (e.g. average spin) ? */
    bool dump_information = false;
    /* Number of CUDA threads to use */
    int thread_count = 64;
    /* Initial configuration of the lattice based on this seed */
    string seed = "COSC3500";
    /* The location to write images of the lattice */
    string output_dir_name = "output_img/";
    FILE* info_dump_file = nullptr;

    /* Parse arguments */
    char opt;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
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
    cout << "Dimension:           " << dimension << "px\n"
         << "Refresh:             " << update_rate << " iterations between each image\n"
         << "Iterations:          " << iterations << "\n"
         << "Threads:             " << thread_count << "\n"
         << "Seed:                " << seed << "\n"
         << "Temperature:         " << temperature << "\n"
         << "Output dir:          " << output_dir_name << "\n"
         << "Adjusted Iterations: " << iterations * dimension * dimension << "\n";


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


    curandGenerator_t curand_generator;
    float* random_device;
    int* lattice_device;
    size_t lattice_sz = lattice->get_dimension() * lattice->get_dimension();
    int block_count = floor(lattice_sz / thread_count) + 1;

    cout << "Block count is " << block_count << endl;
    cout << "Attempting to allocate " 
         << (lattice_sz * sizeof(float) + lattice_sz * sizeof(int)) / 1000000 
         << "MB of VRAM...\n" << endl;

    /* Allocate CUDA memory */
    CUDACheckErr(cudaMalloc(&lattice_device, lattice_sz * sizeof(int)));
    CUDACheckErr(cudaMalloc(&random_device, lattice_sz * sizeof(float)));

    /* Copy the initially randomised lattice to device */
    CUDACheckErr(cudaMemcpy(lattice_device, lattice->lattice, lattice_sz * sizeof(float), 
        cudaMemcpyHostToDevice));

    /* Initialise the CURand generator */
    CUDACheckErr(curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT));
    CUDACheckErr(curandSetPseudoRandomGeneratorSeed(curand_generator, lattice->get_seed()));

    /* Initialise timer */
    _time_point start, end;
    _time_point t0 = _clock::now();
    start = t0;

    for (long i = 1; i <= iterations; i++) {
        /* Populate random floats between 0 and 1 on the device for each lattice site */
        CUDACheckErr(curandGenerateUniform(curand_generator, random_device, lattice_sz));

        /* Perform monte carlo iterations on lattice sites with even parity */
        evaluate_disjoint_component<<<block_count, thread_count>>>(lattice_device, random_device, 
            dimension, temperature, PARITY_EVEN);
        CUDACheckErr(cudaDeviceSynchronize());

        /* Perform monte carlo iterations on lattice sites with odd parity */
        evaluate_disjoint_component<<<block_count, thread_count>>>(lattice_device, random_device, 
            dimension, temperature, PARITY_ODD);
        CUDACheckErr(cudaDeviceSynchronize());
        
        if (i % update_rate == 0) {
            /* Copy the lattice back from the device if required */
            if (graphical || dump_information)
                CUDACheckErr(cudaMemcpy(lattice->lattice, lattice_device, lattice_sz * sizeof(int), 
                    cudaMemcpyDeviceToHost));

            end = _clock::now();
            /* Determine how long it took to execute 'update_rate' iterations */
            print_progress(calculate_time_delta(end, start), i, iterations, update_rate);
            start = _clock::now();

            if (dump_information)
                lattice->dump_information(info_dump_file, i);
            
            if (graphical)
                lattice->write_to_bitmap((output_dir_name + "/outfile_" 
                    + to_string((int)(i / update_rate)) + ".bmp"));
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

__device__
int neighbour_sum(int* lattice, int site_index, int lattice_dimension) {
    int adjacent_sum = 0;

    /* North neighbour, if in bounds */
    adjacent_sum += (site_index >= lattice_dimension ? 
        lattice[site_index - lattice_dimension] : 0);
        
    /* East neighbour, if in bounds */
    adjacent_sum += (site_index + 1 < lattice_dimension * lattice_dimension ? 
        lattice[site_index + 1] : 0);

    /* South neighbour, if in bounds */
    adjacent_sum += (site_index < lattice_dimension * (lattice_dimension - 1) ? 
        lattice[site_index + lattice_dimension] : 0);

    /* West neighbour, if in bounds */
    adjacent_sum += (site_index - 1 >= 0 ? 
        lattice[site_index - 1] : 0);

    return adjacent_sum;
}

/*
 * Run on a CUDA capable device.
 *
 * For each site on the lattice, a CUDA thread will execute this function. Half of 
 * those CUDA threads will return without changing the lattice (depending
 * on the value of 'parity'). For each of the threads that do continue, the 'energy'
 * of adjacent neighbours are calculated and the lattice site managed by this
 * thread is flipped.
 */
__global__
void evaluate_disjoint_component(int* lattice, const float* random, int lattice_dimension,
    float temperature, int parity) {
    /* 
     * This is the index of a specific lattice site. A CUDA thread is
     * run for every lattice site. Half of those threads will return
     * without changing the lattice
     */
    int site_index = blockDim.x  * blockIdx.x + threadIdx.x;

    /* Out of bounds - Thread not needed */
    if (site_index >= lattice_dimension * lattice_dimension)
        return ;

    /* 
     * For odd lattices, lattice sites (threads) with an index not 
     * matching the parity are not considered 
     */
    if (lattice_dimension % 2 != 0 && site_index % 2 == parity)
        return ;

    /*
     * If the dimension of the lattice is even, then the disjoint sets of lattice sites
     * will overlap.
     */
    if (lattice_dimension % 2 == 0) {
        int row = site_index / lattice_dimension;
        
        /* 
         * If the parity is odd:
         *  . If the row is even and site_index is odd, this thread is not needed
         *  . If the row is odd and site_index is even, this thread is not needed
         */
        if (parity == PARITY_ODD && row % 2 != site_index % 2)
            return ;

        /*
         * If the parity is even
         *  . If the row is even and site_index is even, this thread is not needed
         *  . If the row is odd and site_index is odd, this thread is not needed
         */
        if (parity == PARITY_EVEN && row % 2 == site_index % 2)
            return;
    }

    int energy_0, energy_1, energy_change;
    
    energy_0 = -2 * 1 * lattice[site_index] * neighbour_sum(lattice, site_index, lattice_dimension);
    /* Flip the spin */
    lattice[site_index] *= -1;
    
    energy_1 = -2 * 1 * lattice[site_index] * neighbour_sum(lattice, site_index, lattice_dimension);

    energy_change = energy_1 - energy_0;

    /*
     * If Energy_change <= 0, we accept the spin flip performed above
     * If Energy_change >  0, we accept the spin flip above if r <= exp(-energy_change/temperature)
     * Otherwise, we revert the spin.
     */
    if (energy_change > 0 && random[site_index] > exp(-1.0 * energy_change / temperature))
        lattice[site_index] *= -1;
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
 * Displays usage message to stderr 
 */
void usage(char* prog_name) {
    cerr << prog_name << " pattern_file [--dimension (-n)]" 
            << " [--[r]efresh-rate] [--thread-count (-c)]" 
            << " [--[g]raphical] [--[i]terations] [--[s]eed] [--[t]emperature]"
            << " [--[o]utput <output_dir_name>]"
            << " [--[d]ump-information]" << endl;
    exit(1);
}
