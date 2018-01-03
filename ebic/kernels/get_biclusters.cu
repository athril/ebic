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


#ifndef _GET_BICLUSTERS_CU_
#define _GET_BICLUSTERS_CU_

#include "evaluate_trends.cu"

template <typename T>
__global__ void get_biclusters(const int SHARED_MEM_SIZE,
                        const float APPROX_TRENDS_RATIO,
                        const int NEGATIVE_TRENDS_ENABLED,
                        const float EPSILON,
                        int num_biclusters,
                        int *bicl_indices,
                        int size_indices,
                        int *compressed_biclusters,
                        int num_rows,
                        int num_cols,
                        T *data,
                        int *coverage) {
  extern __shared__ int memory[];
  int *trend_increasing=memory;
  int *trend_decreasing=&trend_increasing[SHARED_MEM_SIZE];
  //float *trendvalue=(float*)&trend_decreasing[SHARED_MEM_SIZE];
  T *trendvalue=(T*)&trend_decreasing[SHARED_MEM_SIZE];

  long long int index_x = blockIdx.x * blockDim.x + threadIdx.x;    //block of bicluster
  long long int index_y = blockIdx.y * blockDim.y + threadIdx.y;    //block of row


  evaluate_trends<T>(bicl_indices, compressed_biclusters, num_rows, num_cols, data, trend_increasing, trendvalue, EPSILON);



  if (trend_increasing[threadIdx.y]<APPROX_TRENDS_RATIO*(bicl_indices[index_x+1]-bicl_indices[index_x])) {
    trend_increasing[threadIdx.y]=0;
  } else
    trend_increasing[threadIdx.y]=1;

  if (NEGATIVE_TRENDS_ENABLED) {
    evaluate_trends<T>(bicl_indices, compressed_biclusters, num_rows, num_cols, data, trend_decreasing, trendvalue, EPSILON, -1);
    if (trend_decreasing[threadIdx.y]<APPROX_TRENDS_RATIO*(bicl_indices[index_x+1]-bicl_indices[index_x])) {
      trend_decreasing[threadIdx.y]=0;
    } else {
      trend_decreasing[threadIdx.y]=1;
    }
  }
  __syncthreads();

  if (index_y<num_rows && index_x<num_biclusters) {
    coverage[index_y+num_rows*index_x]=trend_increasing[threadIdx.y]|trend_decreasing[threadIdx.y];
  }
}


#endif