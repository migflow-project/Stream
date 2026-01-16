// 
// Stream - Copyright (C) <2025-2026>
// <Universite catholique de Louvain (UCL), Belgique>
// 
// List of the contributors to the development of Stream: see AUTHORS file.
// Description and complete License: see LICENSE file.
// 
// This file is part of Stream. Stream is free software:
// you can redistribute it and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.
// 
// Stream is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License along with Stream. 
// If not, see <https://www.gnu.org/licenses/>.
// 
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
#include "DirectAlphaShape3D.hpp"
#include "ava_host_array.hpp"
#include "defines.h"
#include "timing.hpp"
#include "primitives.hpp"

const char *argp_program_version = "IMR26-release";
const char *argp_program_bug_address = "nathan.tihon@uclouvain.be";
const char doc[] = "Sample n uniform 3D points and execute our alphashape algorithm on constant alpha";

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
    uint32_t npoints;           
    uint32_t nruns;
    fp_tt alpha;
    int rng;          
    bool output;
};

struct argp_option options[] = {
    {"file",      'i', "path",      0,                   "Path to the coordinate file",          0},
    {"rngseed",   's', "uint",      0,                   "Seed for the Random Number Generator", 0},
    {"numpts",    'n', "uint",      0,                   "Number of points to generate",         1},
    {"alpha",     'a', "float",     0,                   "alpha value for the alpha-shape",      2},
    {"runs",      'r', "int",       0,                   "Number of iterations",                 2},
    {"output",    'o', "",          OPTION_ARG_OPTIONAL, "Whether or not to output results",     3},
    {0,           0,   0,           0,                   0,                                      0}
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    cmd_args_st *args = (cmd_args_st *)state->input;
    switch (key) {
        case 'i':
            args->fname = arg;
            break;
        case 's':
            args->rng = (uint32_t)strtoull(arg, NULL, 10);
            break;
        case 'n':
            args->npoints = (uint32_t)strtoull(arg, NULL, 10);
            break;
        case 'a':
            args->alpha = (fp_tt)strtod(arg, NULL);
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
        .npoints       = 100,
        .nruns         = 30,
        .alpha         = 1.f,
        .rng           = 42u,
        .output        = false
    };
    argp_parse(&argp, argc, argv, 0, 0, (void *)&default_args);
    return default_args;
}

int main(int argc, char **argv) {

    stream::utils::Chrono c;

    // Parse arguments
    cmd_args_st args = parse_args(argc, argv);
    srand(args.rng);
    printf("RNG seed : %u\n", args.rng);


    // ======================== Generate random points in [0, 1]^3 =================
    AvaHostArray<Sphere3D, int>::Ptr h_nodes = AvaHostArray<Sphere3D, int>::create({(int) args.npoints});

    for (uint32_t i = 0; i < args.npoints; i++){
        Vec3f const v = {
            (fp_tt) rand() / RAND_MAX,
            (fp_tt) rand() / RAND_MAX,
            (fp_tt) rand() / RAND_MAX
        };
        h_nodes(i) = {v, args.alpha};
    }

    // ======================== Set nodes =====================================
    c.start();
    stream::mesh::AlphaShape3D alphashape;
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
        edata[aiter].ntriangles = alphashape.n_elems;
    }

    // Print run statistics
    printf("run alpha npoints ntri tlbvh tashape tcompress ttot\n");
    for (uint32_t i = 0; i < args.nruns; i++){
        printf("%u %lf %u %u %lf %lf %lf %lf\n", i, args.alpha, args.npoints, edata[i].ntriangles, edata[i].tlbvh, edata[i].tashape, edata[i].tcompress, edata[i].ttot);
    }
    free(edata);

    // Output point set / triangles
    if (args.output){
        std::vector<stream::mesh::AlphaShape3D::Elem> elems;
        std::vector<Sphere3D> nodes;

        alphashape.getElem(elems);
        alphashape.getCoordsMorton(nodes);

        FILE* felem = fopen("elems.txt", "w");
        for (stream::mesh::AlphaShape3D::Elem const e : elems) {
            fprintf(felem, "%u %u %u %u\n", e.a, e.b, e.c, e.d);
        }
        fclose(felem);

        FILE* fnode = fopen("nodes.txt", "w");
        for (Sphere3D const s : nodes) {
            fprintf(fnode, "%.5f %.5f %.5f\n", s.c[0], s.c[1], s.c[2]);
        }
        fclose(fnode);
    }

    return EXIT_SUCCESS;
}
