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



#ifndef _EVOBIC_CU_
#define _EVOBIC_CU_

#include "evobic.hxx"
#include "kernels/get_biclusters.cu"
#include "kernels/calculate_fitness.cu"


#define gpuCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != 0) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),file,line);
    if (abort)
      exit(code);
  }
}




EvoBic::EvoBic(int num_gpus, int cols, int rows, float approx_trend_ratio, int negative_trends, problem_t *problem) : NUM_GPUs(num_gpus), NUM_COLUMNS(cols), NUM_ROWS(rows), APPROX_TRENDS_RATIO(approx_trend_ratio), NEGATIVE_TRENDS_ENABLED(negative_trends) {
  gpu_args = new arguments[NUM_GPUs];
  fitness_array=new int*[NUM_GPUs];
  coverage=new int*[NUM_GPUs];
  SHARED_MEMORY_SIZE= pow(2,1+floor(MIN(log(NUM_ROWS),9)));
  const int BLOCKSIZE[2]={1, SHARED_MEMORY_SIZE};
  int remaining_rows=problem->num_rows;
  for (int dev=0; dev<NUM_GPUs; ++dev) {
    gpu_args[dev].dev_data_split=MIN(ceil((double)problem->num_rows/(double)NUM_GPUs), remaining_rows);//problem->num_rows/NUM_GPUs;
    gpu_args[dev].dev_data_pos=dev*ceil((double)problem->num_rows/(double)NUM_GPUs);
    remaining_rows-=gpu_args[dev].dev_data_split;
    //cout << "Splitting data of " << problem->num_rows<< " rows: device " << dev <<": from "<< gpu_args[dev].dev_data_pos << " includes " << gpu_args[dev].dev_data_split << " samples. Fitness starts from " <<  (int)ceil((double)gpu_args[dev].dev_data_pos/(double)BLOCKSIZE[1])*problem->num_biclusters << " with " << (int)ceil((double)gpu_args[dev].dev_data_split/(double)BLOCKSIZE[1])*problem->num_biclusters<< endl;
  }
  for (int dev=0; dev<NUM_GPUs; ++dev) {
    cudaMallocHost((void**)&fitness_array[dev], (int)ceil((double)gpu_args[dev].dev_data_split/(double)BLOCKSIZE[1])*problem->num_biclusters*sizeof(int) );
    cudaMallocHost((void**)&coverage[dev], problem->num_biclusters*gpu_args[dev].dev_data_split*sizeof(int) );
  }

  //cudaMallocHost((void**)&coverage, problem->num_biclusters*problem->num_rows*sizeof(int) );


  #pragma omp parallel num_threads(NUM_GPUs)
  {
    const int dev = omp_get_thread_num(); 
    cudaSetDevice(dev);
    gpuCheck( cudaMalloc((void**)&gpu_args[dev].dev_data, gpu_args[dev].dev_data_split * problem->num_cols * sizeof(float)) );
    gpuCheck( cudaMalloc((void**)&gpu_args[dev].dev_coverage, gpu_args[dev].dev_data_split * problem->num_biclusters * sizeof(int)) );  
    gpuCheck( cudaMalloc((void**)&gpu_args[dev].dev_fitness_array, (int)ceil((double)gpu_args[dev].dev_data_split/(double)BLOCKSIZE[1])*problem->num_biclusters * sizeof(int)) );
    gpuCheck( cudaMemcpyAsync(gpu_args[dev].dev_data, problem->data+gpu_args[dev].dev_data_pos*problem->num_cols, gpu_args[dev].dev_data_split*problem->num_cols*sizeof(float), cudaMemcpyHostToDevice ) );
  }

}


EvoBic::~EvoBic() {
  #pragma omp parallel num_threads(NUM_GPUs)
  {
    const int dev = omp_get_thread_num(); 
    cudaSetDevice(dev);
    //cudaFree(gpu_args[dev].dev_data);
    cudaFree(gpu_args[dev].dev_fitness_array);
    cudaFree(gpu_args[dev].dev_coverage);
  }  
  //cudaFree(coverage);
  cudaDeviceReset() ;
}

void EvoBic::get_final_biclusters(problem_t *problem) {
  #pragma omp parallel num_threads(NUM_GPUs) 
  {
    const int dev = omp_get_thread_num(); 
    cudaSetDevice(dev);
    gpuCheck( cudaMalloc((void**)&gpu_args[dev].dev_bicl_indices, (problem->num_biclusters+1) * sizeof(int)) );
    gpuCheck( cudaMalloc((void**)&gpu_args[dev].dev_compressed_biclusters, problem->size_indices * sizeof(int)) );    
    gpuCheck( cudaMemcpyAsync(gpu_args[dev].dev_bicl_indices, problem->bicl_indices, (problem->num_biclusters+1)*sizeof(int), cudaMemcpyHostToDevice) );
    gpuCheck( cudaMemcpyAsync(gpu_args[dev].dev_compressed_biclusters, problem->compressed_biclusters, problem->size_indices*sizeof(int), cudaMemcpyHostToDevice) );
  }

  #pragma omp parallel num_threads(NUM_GPUs) 
  {
    const int dev = omp_get_thread_num(); 
    cudaSetDevice(dev);
    const int BLOCKSIZE[2]={1, SHARED_MEMORY_SIZE};
    dim3 dimBlock(BLOCKSIZE[0],BLOCKSIZE[1]); //[biclusters, rows]
    dim3 dimGrid( problem->num_biclusters, gpu_args[dev].dev_data_split);
    get_biclusters<float><<< dimGrid, dimBlock,SHARED_MEMORY_SIZE*(2*sizeof(int)+sizeof(float)) >>> (SHARED_MEMORY_SIZE, APPROX_TRENDS_RATIO, NEGATIVE_TRENDS_ENABLED, EPSILON, problem->num_biclusters, gpu_args[dev].dev_bicl_indices, problem->size_indices, gpu_args[dev].dev_compressed_biclusters, gpu_args[dev].dev_data_split, problem->num_cols, gpu_args[dev].dev_data, gpu_args[dev].dev_coverage);
    gpuCheck( cudaMemcpy(coverage[dev], gpu_args[dev].dev_coverage, gpu_args[dev].dev_data_split*problem->num_biclusters*sizeof(int), cudaMemcpyDeviceToHost ) );

  }
  gpuCheck( cudaDeviceSynchronize() );
}


void EvoBic::print_biclusters_synthformat(problem_t *problem, string results_filename) {
  std::ofstream results_file;
  results_file.open(results_filename.c_str());
  results_file << "#" << problem->num_biclusters << endl;
  for (int i=0; i<problem->num_biclusters; ++i) {
    results_file << "Bicluster([";
    bool is_first=true;
    for (int dev=0; dev<NUM_GPUs; ++dev) {
      for (int j=0; j<gpu_args[dev].dev_data_split; j++) {
        if (coverage[dev][i*gpu_args[dev].dev_data_split+j]) {
          if (is_first) {
            is_first=false;
          } else 
            results_file <<", ";
          results_file << dev*gpu_args[dev].dev_data_pos+j;
        }
      }
    }
    results_file << "], [";
    for(int compressedId=problem->bicl_indices[i]; compressedId<problem->bicl_indices[i+1]; ++compressedId) {
      if (compressedId!=problem->bicl_indices[i])
        results_file << ", ";
      results_file << problem->compressed_biclusters[compressedId];
    }
    results_file << "])" << endl;
  }
  results_file.close();
}

void EvoBic::print_biclusters_blocks(problem_t *problem, string results_filename) {
  std::ofstream results_file;
  results_file.open(results_filename.c_str());
  for (int i=0; i<problem->num_biclusters; ++i) {
    list<int> row_indices, col_indices;
    std::string bicluster_id = std::string(4 - to_string(i).length(), '0') + to_string(i);


    for (int dev=0; dev<NUM_GPUs; ++dev) {
      for (int j=0; j<gpu_args[dev].dev_data_split; j++) {
        if (coverage[dev][i*gpu_args[dev].dev_data_split+j]) {
          row_indices.push_back(dev*gpu_args[dev].dev_data_pos+j);
        }
      }
    }


    for(int compressedId=problem->bicl_indices[i]; compressedId<problem->bicl_indices[i+1]; ++compressedId) {
      col_indices.push_back(problem->compressed_biclusters[compressedId]);
    }

    results_file << "BC" << bicluster_id ;
    results_file << " [" << row_indices.size() << " x " << col_indices.size() <<"]" << endl;
    results_file << "Rows/Columns";
    for (auto idc : col_indices) {
        results_file << "\t";
        results_file << problem->col_headers[idc];
    }
    results_file << endl;
    for (auto idr : row_indices) {
      results_file << problem->row_headers[idr];
      for (auto idc : col_indices) {
        results_file << "\t";
        results_file << problem->data[idc+NUM_COLUMNS*idr];
      }
      results_file << endl;
    }
    results_file << endl << endl;
    row_indices.clear();
    col_indices.clear();
  }
  results_file.close();

}

void EvoBic::print_biclusters(problem_t *problem, string results_filename) {
  std::ofstream results_file;
  results_file.open(results_filename.c_str());

  for (int i=0; i<problem->num_biclusters; ++i) {
    list<int> indices;
    std::string bicluster_id = std::string(4 - to_string(i).length(), '0') + to_string(i);
    results_file << "BC" << bicluster_id << endl;


    for (int dev=0; dev<NUM_GPUs; ++dev) {
      for (int j=0; j<gpu_args[dev].dev_data_split; j++) {
        if (coverage[dev][i*gpu_args[dev].dev_data_split+j]) {
          indices.push_back(dev*gpu_args[dev].dev_data_pos+j);
        }
      }
    }
    indices.sort();
    results_file << "Genes [" << indices.size() << "]:";
    for (auto id : indices) {
      results_file << " ";
      results_file << "g" << id;
    }

    indices.clear();
    results_file << endl;

    for(int compressedId=problem->bicl_indices[i]; compressedId<problem->bicl_indices[i+1]; ++compressedId) {
      indices.push_back(problem->compressed_biclusters[compressedId]);
    }

    indices.sort();
    results_file << "Conds [" << indices.size() << "]:";
    for (auto id : indices) {
      results_file << " ";
      results_file << "c" << id;
    }
    results_file << endl << endl;
    indices.clear();
  }
  results_file.close();
}



void EvoBic::determine_fitness(problem_t *problem, float *fitness) {
  const int BLOCKSIZE[2]={1, SHARED_MEMORY_SIZE};
//  assert(BLOCKSIZE[0]*BLOCKSIZE[1]<=1024);
  assert(problem->num_rows > BLOCKSIZE[1]);
  omp_set_num_threads(NUM_GPUs); 
    
  for (int j=0; j<problem->num_biclusters; ++j) {
     fitness[j]=0;
  }

  //cout << "GPU: SIZE INDICES" << problem->size_indices << endl;
  #pragma omp parallel num_threads(NUM_GPUs) 
  {
    const int dev = omp_get_thread_num(); 
    cudaSetDevice(dev);
    gpuCheck( cudaMalloc((void**)&gpu_args[dev].dev_bicl_indices, (problem->num_biclusters+1) * sizeof(int)) );
    gpuCheck( cudaMalloc((void**)&gpu_args[dev].dev_compressed_biclusters, problem->size_indices * sizeof(int)) );    
    gpuCheck( cudaMemcpyAsync(gpu_args[dev].dev_bicl_indices, problem->bicl_indices, (problem->num_biclusters+1)*sizeof(int), cudaMemcpyHostToDevice) );
    gpuCheck( cudaMemcpyAsync(gpu_args[dev].dev_compressed_biclusters, problem->compressed_biclusters, problem->size_indices*sizeof(int), cudaMemcpyHostToDevice) );
  }

  #pragma omp parallel num_threads(NUM_GPUs) 
  {
    const int dev = omp_get_thread_num(); 
    cudaSetDevice(dev);
    dim3 dimBlock(BLOCKSIZE[0],BLOCKSIZE[1]); //[biclusters, rows]
    dim3 dimGrid( problem->num_biclusters, ceil((double)gpu_args[dev].dev_data_split/(double)BLOCKSIZE[1]));
    calculate_fitness<float><<< dimGrid, dimBlock, SHARED_MEMORY_SIZE*(sizeof(float)+sizeof(int)) >>> (SHARED_MEMORY_SIZE, EPSILON, gpu_args[dev].dev_bicl_indices, problem->num_biclusters, gpu_args[dev].dev_compressed_biclusters, gpu_args[dev].dev_data_split, problem->num_cols, gpu_args[dev].dev_data, gpu_args[dev].dev_fitness_array);
  }

  #pragma omp parallel num_threads(NUM_GPUs) 
  {
    const int dev = omp_get_thread_num(); 
    cudaSetDevice(dev);
    gpuCheck( cudaMemcpy(fitness_array[dev], gpu_args[dev].dev_fitness_array, (int)ceil((double)gpu_args[dev].dev_data_split/(double)BLOCKSIZE[1])*problem->num_biclusters*sizeof(int), cudaMemcpyDeviceToHost) );
  }
  gpuCheck( cudaDeviceSynchronize() );
  //printf("\n-----------------------------------------------------------------------------------\n");
  
  
  for (int i=0; i<problem->num_biclusters; ++i) {
    for (int dev=0; dev<NUM_GPUs; ++dev) {
      for (int j=0; j<(int)ceil((double)gpu_args[dev].dev_data_split/(double)BLOCKSIZE[1]); ++j) {
        fitness[i]+=fitness_array[dev][problem->num_biclusters*j+i];
      }
    }
    //printf("[%3.2f x %d =", fitness[i], problem->bicl_indices[i+1]-problem->bicl_indices[i]);
    fitness[i]=(fitness[i]>0 ? pow(2,MIN(fitness[i]-MIN_NO_ROWS,0))*(problem->bicl_indices[i+1]-problem->bicl_indices[i])*(fitness[i]>1 ? log2( MAX(fitness[i]-1,0.0)) : 0) : 0);
    //printf("%3.2f]\n", fitness[i]);
  }

  #pragma omp parallel num_threads(NUM_GPUs) 
  {
    const int dev = omp_get_thread_num(); 
    cudaSetDevice(dev);
    cudaFree(gpu_args[dev].dev_bicl_indices);
    cudaFree(gpu_args[dev].dev_compressed_biclusters);
  }
}



#endif