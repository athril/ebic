/***

Copyright (c) 2017 Patryk Orzechowski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

***/


#ifndef _EVOBIC_H_
#define _EVOBIC_H_

#include <assert.h>
#include <cuda.h>
#include <math.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include "dataIO.hxx"
#include "parameters.hxx"

class EvoBic {
  const int NUM_COLUMNS;
  const int NUM_ROWS;
  const int NUM_GPUs;
  int SHARED_MEMORY_SIZE;
  float APPROX_TRENDS_RATIO;
  int NEGATIVE_TRENDS_ENABLED;
  int **fitness_array, **coverage;

struct arguments {
  int *dev_bicl_indices;
  int *dev_compressed_biclusters;
  float *dev_data;
  int dev_data_pos;
  int dev_data_split;
  int *dev_coverage;
  int *dev_fitness_array;
} *gpu_args;



public:
    EvoBic(int num_gpus, int cols, int rows, float approx_trends_ratio, int negative_trends, problem_t *problem);
    void get_final_biclusters(problem_t *problem);
    void print_biclusters(problem_t *problem, string results_filename);
    void print_biclusters_blocks(problem_t *problem, string results_filename);
    void print_biclusters_synthformat(problem_t *problem, string results_filename);

    void determine_fitness(problem_t *problem, float *fitness);

    ~EvoBic();
};

#endif