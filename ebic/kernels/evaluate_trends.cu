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
                        float MISSING_VALUE,
                        int increasing=1) {
  long long int index_x = blockIdx.x * blockDim.x + threadIdx.x;    //block of bicluster
  long long int index_y = blockIdx.y * blockDim.y + threadIdx.y;    //block of row

  trendcheck[threadIdx.x]=0;
  trendvalue[threadIdx.x]=0;

  if (index_x<num_rows ) {
      trendcheck[threadIdx.x] = 1;
      trendvalue[threadIdx.x] = data[compressed_biclusters[bicl_indices[index_y]]+num_cols*index_x];
  }
  __syncthreads();
  if (index_x<num_rows ) {
      for(int compressedId=bicl_indices[index_y]+1; compressedId<bicl_indices[index_y+1]; ++compressedId) {
        int pos=compressed_biclusters[compressedId];
        trendcheck[threadIdx.x] += (increasing*(data[pos+num_cols*index_x]+EPSILON-trendvalue[threadIdx.x])>= 0 && data[pos+num_cols*index_x]!=MISSING_VALUE);
        trendvalue[threadIdx.x] = data[pos+num_cols*index_x];
        __syncthreads();
      }
  }
}

#endif