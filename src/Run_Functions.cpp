//
// Created by marcuskeil on 05/08/25.
//
#import "../include/Run_Functions.hpp"


std::tuple<bool, bool, bool, bool, bool, int, int, float> retrieve_arguments(int argc, char **argv) {
    bool Mem_Run = false;
    bool Plot = false;
    bool FFTW = false;
    bool cuFFT = false;
    bool rocFFT = false;
    int run_count = 6;
    int repeat_count = 5;
    float mem_start = 1000; // in [MB]
    opterr = 0;

    while ( true ) {
        static option long_options[] = {
            {"Memory Count Run Bool",no_argument, nullptr,'e'},
            {"Plot Bool",no_argument, nullptr,'p'},
            {"FFTW Bool",no_argument, nullptr,'f'},
            {"NVIDIA-cuFFT Bool",no_argument, nullptr,'n'},
            {"AMD-rocFFT Bool",no_argument, nullptr,'a'},
            {"Number of runs to perform beyond first",optional_argument,nullptr, 'r'},
            {"Number of times to repeat runs, for averages",optional_argument,nullptr, 'c'},
            {"Starting memory size if Memory Run",optional_argument, nullptr, 's'},
            {nullptr, 0, nullptr, 0}
        };
        /* getopt_long stores the option index here. */
        int option_index = 0;

        const int option = getopt_long(argc, argv, "epfnar:c:s:", long_options, &option_index);

        /* Detect the end of the options. */
        if (option == -1)
            break;
        switch (option) {
            case 'e':
                Mem_Run = true;
            case 'p':
                Plot = true;
                break;
            case 'f':
                FFTW = true;
                break;
            case 'n':
                cuFFT = true;
                break;
            case 'a':
                rocFFT = true;
                break;
            case 'r':
                run_count = static_cast<int>(strtod(optarg, nullptr));
                break;
            case 'c':
                repeat_count = static_cast<int>(strtod(optarg, nullptr));
                assert(run_count >= 1);
                assert(run_count< 7);
                break;
            case 's':
                mem_start = float(strtod(optarg, nullptr));
                break;
            case '?':
                if (optopt == 'c')
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint(optopt))
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf(stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
                break;
            default:
                abort();
        }
    }
    return std::make_tuple(Mem_Run, Plot, FFTW, cuFFT, rocFFT, run_count, repeat_count, mem_start);
}

std::vector<int> get_elements(const int run_count) {
    std::vector<int> element_counts = {
        31360000, // ~500MB
        63011844, // ~1GB
        125440000, // ~2GB
        248062500, // ~4GB
        486202500, // ~8GB
        992250000, // ~16GB
        2007040000 // ~32GB
    };
    std::vector<int> requested(element_counts.begin(), element_counts.begin() + run_count);
    return requested;
}

std::vector<float> linspace(const float start, const float end, const int count){
    std::vector<float> linspace_vec;
    linspace_vec.push_back(start);
    if (count == 1){
        return linspace_vec;
    } else {
        const float delta = (end - start) / static_cast<float>(count - 1);
        for(int i=1; i < count-1; i++){
            linspace_vec.push_back(start + delta*static_cast<float>(i));
        }
        linspace_vec.push_back(end);
        return linspace_vec;
    }
}

std::vector<float> get_memories(const float start, const int count){
    std::vector<float> double_pace_vec;
    float current = start;
    double_pace_vec.push_back(start);
    for(int i=2; i < count+2; i++){
        current = current*2;
        double_pace_vec.push_back(current);
    }
    return double_pace_vec;
}


