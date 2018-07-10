/***

Copyright (c) 2017-present Patryk Orzechowski

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


#ifndef _EBIC_H_
#define _EBIC_H_

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

void checkAvailableGPUs(int);


class EBic {
  const int num_cols;               // number of columns in the input matrix
  const int num_rows;               // number of rows in the input matrix
  const float* data;                // input matrix
  const int NUM_GPUs;               // determines number of GPUs used by ebic
  int NUM_AVAILABLE_GPUs;           // determines number of devices available
  int SHARED_MEMORY_SIZE;           // stores the size of shared memory
  const float APPROX_TRENDS_RATIO;  // determines what percentage of columns in bicluster should preserve the trend
  const int NEGATIVE_TRENDS_ENABLED;// determines if negatively correlated patterns should be included
  int NUM_TRENDS;                   // number of biclusters
  float MISSING_VALUE;              // stores encoding of a missing value
  int **fitness_array, **coverage;
  vector<string> *row_headers;
  vector<string> *col_headers;
  problem_t problem;




  struct gpu_arguments {
    int *dev_bicl_indices;
    int *dev_compressed_biclusters;
    float *dev_data;
    long long dev_data_pos;
    long long dev_data_split;
    int *dev_coverage;
    int *dev_fitness_array;
  } *gpu_args;



  public:
    EBic(int num_gpus, float approx_trends_ratio, int negative_trends, int trends_population_size, float missing_value, float *data, int rows, int cols, std::vector<string> &row_headers, std::vector<string> &col_headers);

    void get_final_biclusters(problem_t *problem);
    void print_biclusters_r(string results_filename);
    void print_biclusters_blocks(string results_filename);
    void print_biclusters_synthformat(string results_filename);
    void determine_fitness(problem_t *problem, float *fitness);

    ~EBic();
};

#endif