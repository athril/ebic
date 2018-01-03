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


#ifndef _EVALUATE_TRENDS_CU_
#define _EVALUATE_TRENDS_CU_

template<typename T>
__device__ void evaluate_trends(int *bicl_indices,
                        int *compressed_biclusters,
                        int num_rows,
                        int num_cols,
                        T *data,
                        int *trendcheck,
                        float *trendvalue,
                        const float EPSILON,
                        int increasing=1) {
  long long int index_x = blockIdx.x * blockDim.x + threadIdx.x;    //block of bicluster
  long long int index_y = blockIdx.y * blockDim.y + threadIdx.y;    //block of row

  trendcheck[threadIdx.y]=0;
  trendvalue[threadIdx.y]=0;

  if (index_y<num_rows ) {
      trendcheck[threadIdx.y] = 1;
      trendvalue[threadIdx.y] = data[compressed_biclusters[bicl_indices[index_x]]+num_cols*index_y];
  }
  __syncthreads();
  if (index_y<num_rows ) {
      for(int compressedId=bicl_indices[index_x]+1; compressedId<bicl_indices[index_x+1]; ++compressedId) {
        int pos=compressed_biclusters[compressedId];
        trendcheck[threadIdx.y] += (increasing*(data[pos+num_cols*index_y]+EPSILON-trendvalue[threadIdx.y])>= 0);
        trendvalue[threadIdx.y] = data[pos+num_cols*index_y];
        __syncthreads();
      }
  }
}

#endif