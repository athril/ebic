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


#ifndef _CALCULATE_FITNESS_CU_
#define _CALCULATE_FITNESS_CU_

#include "evaluate_trends.cu"

template <typename T>
__global__ void calculate_fitness(int SHARED_MEM_SIZE,
                        const float EPSILON,
                        float MISSING_VALUE,
                        int *bicl_indices,
                        int size_indices,
                        int *compressed_biclusters,
                        int num_rows,
                        int num_cols,
                        T *data,
                        int *fitness_array) {
  long long int index_x = blockIdx.x * blockDim.x + threadIdx.x;    //block of row
  long long int index_y = blockIdx.y * blockDim.y + threadIdx.y;    //block of bicluster

  extern __shared__ int memory[];
  int *trendcheck=memory;
  T *trendvalue = (T*)&trendcheck[SHARED_MEM_SIZE];


  evaluate_trends(bicl_indices, compressed_biclusters, num_rows, num_cols, data, trendcheck, trendvalue, EPSILON, MISSING_VALUE);

  if (trendcheck[threadIdx.x]<(bicl_indices[index_y+1]-bicl_indices[index_y])) {
    trendcheck[threadIdx.x]=0;
  }
  else {
    trendcheck[threadIdx.x]=1;
  }
  __syncthreads();


  for(int offset = blockDim.x/2; offset > 0;offset >>= 1) {
    if(threadIdx.x < offset && index_x<num_rows) {
      trendcheck[threadIdx.x] += trendcheck[threadIdx.x+offset];
    }
    __syncthreads();
  }

  if (threadIdx.x==0 && index_x<num_rows) {
    fitness_array[blockIdx.x*size_indices+index_y]=trendcheck[0];
  }
}


#endif