#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <argp.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ava.h"
#include "DirectAlphaShape2D.hpp"
#include "ava_host_array.hpp"
#include "defines.h"
#include "timing.hpp"
#include "primitives.hpp"

const char *argp_program_version = "IMR26-release";
const char *argp_program_bug_address = "nathan.tihon@uclouvain.be";
const char doc[] = "Takes file with 3 columns : x, y, alpha. Call our algorithm on this point cloud with variable alpha.";

// Struct to store info about each runs
typedef struct exec_data exec_data_st;
struct exec_data {
    uint32_t ntriangles;
    double tlbvh;
    double tashape;
    double tcompress;
    double ttot;
};

// Command line arguments
typedef struct cmd_args_struct cmd_args_st;
struct cmd_args_struct {
    char *fname;
    uint32_t nruns;
    bool output;
};

struct argp_option options[] = {
    {"file",      'i', "path",      0,                   "Path to the file",                     0},
    {"runs",      'r', "int",       0,                   "Number of iterations",                 1},
    {"output",    'o', "",          OPTION_ARG_OPTIONAL, "Whether or not to output results",     2},
    {0,           0,   0,           0,                   0,                                      0}
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    cmd_args_st *args = (cmd_args_st *)state->input;
    switch (key) {
        case 'i':
            args->fname = arg;
            break;
        case 'r':
            args->nruns = (uint32_t)strtoull(arg, NULL, 10);
            break;
        case 'o':
            args->output = true;
            break;

        case ARGP_KEY_ARG:
            {
                argp_usage(state);

                // Add positional arguments by index here
                switch (state->arg_num) {
                    default:
                        break;
                }
            }
        case ARGP_KEY_END:
            // Check if not enough positional args
            break;
        default:
            return ARGP_ERR_UNKNOWN;
    }

    return 0;
}

static struct argp argp = {options, parse_opt, NULL, doc, NULL, NULL, NULL};

cmd_args_st parse_args(int argc, char** argv) {
    cmd_args_st default_args = {
        .fname         = NULL,
        .nruns         = 30,
        .output        = false
    };
    argp_parse(&argp, argc, argv, 0, 0, (void *)&default_args);
    return default_args;
}

int main(int argc, char **argv) {

    stream::utils::Chrono c;

    // Parse arguments
    cmd_args_st args = parse_args(argc, argv);


    // Get total size of file
    FILE* fd = fopen(args.fname, "rb");
    fseek(fd, 0, SEEK_END); // go to end of file
    size_t fsize = ftell(fd); // get offset of end of file
    fseek(fd, 0, SEEK_SET); // go to start of file

    // Read all file at once
    int npoints = (fsize/sizeof(float)) / 3; // /3 because there are 3 columns for each point
    float * data = new float[fsize/sizeof(float)];
    fread(data, 1, fsize, fd);
    fclose(fd);

    // Pack data into sphere2D
    AvaHostArray<Sphere2D, int>::Ptr h_nodes = AvaHostArray<Sphere2D, int>::create({(int) npoints});
    for (int i = 0; i < npoints; i++){
        h_nodes(i) = {{data[3*i], data[3*i+1]}, data[3*i+2]};
    }
    delete [] data;

    // ======================== Set nodes =====================================
    c.start();
    stream::mesh::AlphaShape2D alphashape;
    alphashape.set_nodes(h_nodes);
    c.stop();
    fprintf(stderr, "init %lf ms\n", c.get_ms());

    exec_data_st* edata = (exec_data_st*) calloc(args.nruns, sizeof(exec_data_st));
    for (uint32_t aiter = 0; aiter < args.nruns; aiter++){ 

        c.start();
        alphashape.init();
        c.stop();
        edata[aiter].tlbvh = c.get_ms();

        c.start();
        alphashape.compute();
        c.stop();
        edata[aiter].tashape = c.get_ms();

        c.start();
        alphashape.compress();
        c.stop();
        edata[aiter].tcompress = c.get_ms();

        edata[aiter].ttot = edata[aiter].tashape + edata[aiter].tlbvh;
        edata[aiter].ntriangles = alphashape.n_tri;
    }

    // Print run statistics
    printf("run npoints ntri tlbvh tashape tcompress ttot\n");
    for (uint32_t i = 0; i < args.nruns; i++){
        printf("%u %u %u %lf %lf %lf %lf\n", i, npoints, edata[i].ntriangles, edata[i].tlbvh, edata[i].tashape, edata[i].tcompress, edata[i].ttot);
    }
    free(edata);

    // Output point set / triangles
    if (args.output){
        std::vector<stream::mesh::AlphaShape2D::Elem> elems;
        std::vector<Sphere2D> nodes;

        // Retrieve output
        // WARNING : the triangles use indices of the morton-ordered objects
        //           Hence using the "h_nodes" array will yield nonsensical graphs
        //           To have correct results, retrieve the morton-ordered objects 
        //           using alphashape.getCoordsMorton()
        alphashape.getTri(elems);
        alphashape.getCoordsMorton(nodes);

        FILE* felem = fopen("elems.txt", "w");
        for (stream::mesh::AlphaShape2D::Elem const e : elems) {
            fprintf(felem, "%u %u %u\n", e.a, e.b, e.c);
        }
        fclose(felem);

        FILE* fnode = fopen("nodes.txt", "w");
        for (Sphere2D const s : nodes) {
            fprintf(fnode, "%.5f %.5f\n", s.c[0], s.c[1]);
        }
        fclose(fnode);
    }

    return EXIT_SUCCESS;
}
