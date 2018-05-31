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



#ifndef _EBIC_CU_
#define _EBIC_CU_

#include "ebic.hxx"
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

EBic::EBic(int num_gpus, float approx_trend_ratio, int negative_trends, int num_trends, float missing_value, float *data, int rows, int cols, std::vector<string> &row_headers, std::vector<string> &col_headers)
      : NUM_GPUs(num_gpus), APPROX_TRENDS_RATIO(approx_trend_ratio), NEGATIVE_TRENDS_ENABLED(negative_trends), MISSING_VALUE(missing_value), NUM_TRENDS(num_trends), num_cols(cols), num_rows(rows), data(data) {
  cudaGetDeviceCount(&NUM_AVAILABLE_GPUs);
  assert(NUM_AVAILABLE_GPUs>=NUM_GPUs);
  this->row_headers=&row_headers;
  this->col_headers=&col_headers;
  gpu_args = new gpu_arguments[NUM_GPUs];
  fitness_array=new int*[NUM_GPUs];
  coverage=new int*[NUM_GPUs];

  SHARED_MEMORY_SIZE= pow(2,1+floor(MIN(log(num_rows),9)));

  const int BLOCKSIZE[2]={SHARED_MEMORY_SIZE,1};

  int remaining_rows=num_rows;
  for (int dev=0; dev<NUM_GPUs; ++dev) {
    gpu_args[dev].dev_data_split=MIN(ceil((double)this->num_rows/(double)NUM_GPUs), remaining_rows);//problem->num_rows/NUM_GPUs;
    gpu_args[dev].dev_data_pos=dev*gpu_args[dev].dev_data_split;
    remaining_rows-=gpu_args[dev].dev_data_split;
//    cout << "Splitting data of " << this->num_rows<< " rows: device " << dev <<": from "<< gpu_args[dev].dev_data_pos << " includes " << gpu_args[dev].dev_data_split << " samples. Fitness starts from " <<  (int)ceil((double)gpu_args[dev].dev_data_pos/(double)BLOCKSIZE[0])*num_trends << " with " << (int)ceil((double)gpu_args[dev].dev_data_split/(double)BLOCKSIZE[0])*num_trends<< endl;
  }

  for (int dev=0; dev<NUM_GPUs; ++dev) {
    gpuCheck( cudaMallocHost((void**)&fitness_array[dev], (int)ceil((double)gpu_args[dev].dev_data_split/(double)BLOCKSIZE[0])*this->NUM_TRENDS*sizeof(int) ) );
    gpuCheck( cudaMallocHost((void**)&coverage[dev], gpu_args[dev].dev_data_split*this->NUM_TRENDS*sizeof(int) ) );
  }


//  cout << "AAA:" << gpu_args[0].dev_data_pos*this->num_cols << " " << gpu_args[0].dev_data_split*this->num_cols*sizeof(float) << " " << (int)ceil((double)gpu_args[0].dev_data_split/(double)BLOCKSIZE[0])* this->NUM_TRENDS * sizeof(int) << endl;
//  cout << endl << *(data+gpu_args[0].dev_data_pos*this->num_cols-1) << endl;
  #pragma omp parallel num_threads(NUM_GPUs)
  {
    const int dev = omp_get_thread_num(); 
    cudaSetDevice(NUM_AVAILABLE_GPUs-dev-1);
    gpuCheck( cudaMalloc((void**)&gpu_args[dev].dev_data, gpu_args[dev].dev_data_split * this->num_cols * sizeof(float)) );
    gpuCheck( cudaMalloc((void**)&gpu_args[dev].dev_coverage, gpu_args[dev].dev_data_split * this->NUM_TRENDS * sizeof(int)) );  
    gpuCheck( cudaMalloc((void**)&gpu_args[dev].dev_fitness_array, (int)ceil((double)gpu_args[dev].dev_data_split/(double)BLOCKSIZE[0])* this->NUM_TRENDS * sizeof(int)) );
    gpuCheck( cudaMemcpyAsync(gpu_args[dev].dev_data, data+gpu_args[dev].dev_data_pos*this->num_cols, gpu_args[dev].dev_data_split*this->num_cols*sizeof(float), cudaMemcpyHostToDevice ) );
  }
}



EBic::~EBic() {
  #pragma omp parallel num_threads(NUM_GPUs)
  {
    const int dev = omp_get_thread_num(); 
    cudaSetDevice(NUM_AVAILABLE_GPUs-dev-1);
    cudaFree(gpu_args[dev].dev_data);
    cudaFree(gpu_args[dev].dev_fitness_array);
    cudaFree(gpu_args[dev].dev_coverage);
  }
  cudaDeviceReset() ;
}


void EBic::determine_fitness(problem_t *problem, float *fitness) {
  gpuCheck( cudaDeviceSynchronize() );
  const int BLOCKSIZE[2]={SHARED_MEMORY_SIZE, 1};
  assert(this->num_rows > BLOCKSIZE[0]);
  omp_set_num_threads(NUM_GPUs);

  for (int j=0; j<problem->num_trends; ++j) {
     fitness[j]=0;
  }

  #pragma omp parallel num_threads(NUM_GPUs) 
  {
    const int dev = omp_get_thread_num(); 
    cudaSetDevice(NUM_AVAILABLE_GPUs-dev-1);
    gpuCheck( cudaMalloc((void**)&gpu_args[dev].dev_bicl_indices, (problem->num_trends+1) * sizeof(int)) );
    gpuCheck( cudaMalloc((void**)&gpu_args[dev].dev_compressed_biclusters, problem->size_indices * sizeof(int)) );
    gpuCheck( cudaMemcpyAsync(gpu_args[dev].dev_bicl_indices, problem->bicl_indices, (problem->num_trends+1)*sizeof(int), cudaMemcpyHostToDevice) );
    gpuCheck( cudaMemcpyAsync(gpu_args[dev].dev_compressed_biclusters, problem->compressed_biclusters, problem->size_indices*sizeof(int), cudaMemcpyHostToDevice) );
  }

  #pragma omp parallel num_threads(NUM_GPUs) 
  {
    const int dev = omp_get_thread_num(); 
    cudaSetDevice(NUM_AVAILABLE_GPUs-dev-1);
    dim3 dimBlock(BLOCKSIZE[0],BLOCKSIZE[1]); //[biclusters, rows]
    dim3 dimGrid( ceil((double)gpu_args[dev].dev_data_split/(double)BLOCKSIZE[0]), problem->num_trends);
    //calculate_fitness<float><<< dimGrid, dimBlock, SHARED_MEMORY_SIZE*(sizeof(float)+sizeof(int)) >>> (SHARED_MEMORY_SIZE, EPSILON, gpu_args[dev].dev_bicl_indices, problem->num_trends, gpu_args[dev].dev_compressed_biclusters, gpu_args[dev].dev_data_split, this->num_cols, gpu_args[dev].dev_data, gpu_args[dev].dev_fitness_array);
    calculate_fitness<float><<< dimGrid, dimBlock, SHARED_MEMORY_SIZE*(sizeof(float)+sizeof(int)) >>> (SHARED_MEMORY_SIZE, EPSILON, MISSING_VALUE, gpu_args[dev].dev_bicl_indices, problem->num_trends, gpu_args[dev].dev_compressed_biclusters, gpu_args[dev].dev_data_split, this->num_cols, gpu_args[dev].dev_data, gpu_args[dev].dev_fitness_array);
  }

  #pragma omp parallel num_threads(NUM_GPUs) 
  {
    const int dev = omp_get_thread_num(); 
    cudaSetDevice(NUM_AVAILABLE_GPUs-dev-1);
    gpuCheck( cudaMemcpy(fitness_array[dev], gpu_args[dev].dev_fitness_array, (int)ceil((double)gpu_args[dev].dev_data_split/(double)BLOCKSIZE[0])*problem->num_trends*sizeof(int), cudaMemcpyDeviceToHost) );
  }
  gpuCheck( cudaDeviceSynchronize() );
  //printf("\n-----------------------------------------------------------------------------------\n");
  
  
  for (int i=0; i<problem->num_trends; ++i) {
    for (int dev=0; dev<NUM_GPUs; ++dev) {
      for (int j=0; j<(int)ceil((double)gpu_args[dev].dev_data_split/(double)BLOCKSIZE[0]); ++j) {
        fitness[i]+=fitness_array[dev][problem->num_trends*j+i];
//        fitness[i]+=fitness_array[dev][(int)ceil((double)gpu_args[dev].dev_data_split/(double)BLOCKSIZE[0])*i+j];
      }
    }
//    printf("[%3.2f x %d =", fitness[i], problem->bicl_indices[i+1]-problem->bicl_indices[i]);
    fitness[i]=(fitness[i]>0 ? pow(2,MIN(fitness[i]-MIN_NO_ROWS,0))*(problem->bicl_indices[i+1]-problem->bicl_indices[i])*(fitness[i]>1 ? log2( MAX(fitness[i]-1,0.0)) : 0) : 0);
//    printf("%3.2f]\n", fitness[i]);
  }

  #pragma omp parallel num_threads(NUM_GPUs) 
  {
    const int dev = omp_get_thread_num(); 
    cudaSetDevice(NUM_AVAILABLE_GPUs-dev-1);
    cudaFree(gpu_args[dev].dev_bicl_indices);
    cudaFree(gpu_args[dev].dev_compressed_biclusters);
  }
}




void EBic::get_final_biclusters(problem_t *problem) {
  gpuCheck( cudaDeviceSynchronize() );
  omp_set_num_threads(NUM_GPUs);
  this->problem = {problem->num_trends, problem->bicl_indices, problem->size_indices, problem->compressed_biclusters};

  #pragma omp parallel num_threads(NUM_GPUs)
  {
    const int dev = omp_get_thread_num();
    gpuCheck( cudaSetDevice(NUM_AVAILABLE_GPUs-dev-1));
    gpuCheck( cudaMalloc((void**)&gpu_args[dev].dev_bicl_indices, (problem->num_trends+1) * sizeof(int)) );
    gpuCheck( cudaMalloc((void**)&gpu_args[dev].dev_compressed_biclusters, problem->size_indices * sizeof(int)) );
    gpuCheck( cudaMemcpyAsync(gpu_args[dev].dev_bicl_indices, problem->bicl_indices, (problem->num_trends+1)*sizeof(int), cudaMemcpyHostToDevice) );
    gpuCheck( cudaMemcpyAsync(gpu_args[dev].dev_compressed_biclusters, problem->compressed_biclusters, problem->size_indices*sizeof(int), cudaMemcpyHostToDevice) );
  }


  //cout << "SHMEM" << SHARED_MEMORY_SIZE << " " << SHARED_MEMORY_SIZE *(2*sizeof(int)+sizeof(float)) << " "<<  problem->num_trends << " " <<  gpu_args[0].dev_data_split << " " << gpu_args[1].dev_data_split << endl;
  #pragma omp parallel num_threads(NUM_GPUs)
  {
    const int dev = omp_get_thread_num();
    cudaSetDevice(NUM_AVAILABLE_GPUs-dev-1);
    const int BLOCKSIZE[2]={SHARED_MEMORY_SIZE, 1};
    dim3 dimBlock(BLOCKSIZE[0],BLOCKSIZE[1]); //[biclusters, rows]
    dim3 dimGrid( gpu_args[dev].dev_data_split, problem->num_trends);


    get_biclusters<float><<< dimGrid, dimBlock,SHARED_MEMORY_SIZE*(2*sizeof(int)+sizeof(float)) >>> (SHARED_MEMORY_SIZE, APPROX_TRENDS_RATIO, NEGATIVE_TRENDS_ENABLED, EPSILON, MISSING_VALUE, problem->num_trends, gpu_args[dev].dev_bicl_indices, problem->size_indices, gpu_args[dev].dev_compressed_biclusters, gpu_args[dev].dev_data_split, this->num_cols, gpu_args[dev].dev_data, gpu_args[dev].dev_coverage);

    gpuCheck( cudaMemcpy(coverage[dev], gpu_args[dev].dev_coverage, gpu_args[dev].dev_data_split*problem->num_trends*sizeof(int), cudaMemcpyDeviceToHost ) );
    gpuCheck( cudaDeviceSynchronize() );
  }
}






void EBic::print_biclusters_synthformat(string results_filename) {
  std::ofstream results_file;
  results_file.open(results_filename.c_str());
  results_file << "#" << this->problem.num_trends << endl;
  for (int i=0; i<this->problem.num_trends; ++i) {
    results_file << "Bicluster([";
    bool is_first=true;
    for (int dev=0; dev<NUM_GPUs; ++dev) {
      for (int j=0; j<gpu_args[dev].dev_data_split; j++) {
        if (coverage[dev][i*gpu_args[dev].dev_data_split+j]) {
          if (is_first) {
            is_first=false;
          } else 
            results_file <<", ";
          results_file << dev*gpu_args[dev].dev_data_split+j;
        }
      }
    }
    results_file << "], [";
    for(int compressedId=this->problem.bicl_indices[i]; compressedId<this->problem.bicl_indices[i+1]; ++compressedId) {
      if (compressedId!=this->problem.bicl_indices[i])
        results_file << ", ";
      results_file << this->problem.compressed_biclusters[compressedId];
    }
    results_file << "])" << endl;
  }
  results_file.close();
}



void EBic::print_biclusters_blocks(string results_filename) {
  std::ofstream results_file;
  results_file.open(results_filename.c_str());
  for (int i=0; i<this->problem.num_trends; ++i) {
    list<int> row_indices, col_indices;
    std::string bicluster_id = std::string(4 - to_string(i).length(), '0') + to_string(i);


    for (int dev=0; dev<NUM_GPUs; ++dev) {
      for (int j=0; j<gpu_args[dev].dev_data_split; j++) {
        if (coverage[dev][i*gpu_args[dev].dev_data_split+j]) {
          row_indices.push_back(dev*gpu_args[dev].dev_data_pos+j);
        }
      }
    }


    for(int compressedId=this->problem.bicl_indices[i]; compressedId<this->problem.bicl_indices[i+1]; ++compressedId) {
      col_indices.push_back(this->problem.compressed_biclusters[compressedId]);
    }

    results_file << "BC" << bicluster_id ;
    results_file << " [" << row_indices.size() << " x " << col_indices.size() <<"]" << endl;
    results_file << "Rows/Columns";
    for (auto idc : col_indices) {
        results_file << "\t";
        results_file << (*col_headers)[idc];
    }
    results_file << endl;
    for (auto idr : row_indices) {
      results_file << (*row_headers)[idr];
      for (auto idc : col_indices) {
        results_file << "\t";
        results_file << this->data[idc+num_cols*idr];

      }
      results_file << endl;
    }
    results_file << endl << endl;
    row_indices.clear();
    col_indices.clear();
  }
  results_file.close();

}


/*
void EBic::print_biclusters(string results_filename) {
  std::ofstream results_file;
  results_file.open(results_filename.c_str());
  cout << this->problem.num_trends << endl;
  for (int i=0; i<this->problem.num_trends; ++i) {
    list<int> indices;
    std::string bicluster_id = std::string(4 - to_string(i).length(), '0') + to_string(i);
    results_file << "BC" << bicluster_id << endl;


    for (int dev=0; dev<NUM_GPUs; ++dev) {
      for (int j=0; j<gpu_args[dev].dev_data_split; j++) {
        if (coverage[dev][i*gpu_args[dev].dev_data_split+j]) {
          indices.push_back(dev*gpu_args[dev].dev_data_split+j);
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

    for(int compressedId=this->problem.bicl_indices[i]; compressedId<this->problem.bicl_indices[i+1]; ++compressedId) {
      indices.push_back(this->problem.compressed_biclusters[compressedId]);
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
*/


void EBic::print_biclusters_r(string results_filename) {
  std::ofstream results_file;
  results_file.open(results_filename.c_str());
  results_file << "EBIC" << endl;
  for (int i=0; i<this->problem.num_trends; ++i) {
    list<int> row_indices, col_indices;

    for (int dev=0; dev<NUM_GPUs; ++dev) {
      for (int j=0; j<gpu_args[dev].dev_data_split; j++) {
        if (coverage[dev][i*gpu_args[dev].dev_data_split+j]) {
          row_indices.push_back(dev*gpu_args[dev].dev_data_split+j);
        }
      }
    }

    for(int compressedId=this->problem.bicl_indices[i]; compressedId<this->problem.bicl_indices[i+1]; ++compressedId) {
      col_indices.push_back(this->problem.compressed_biclusters[compressedId]);
    }

    results_file << row_indices.size() << " " << col_indices.size() << endl;

    for (auto idr : row_indices) {
      results_file << (*row_headers)[idr];
      results_file << " ";
    }
    results_file << endl;

    for (auto idc : col_indices) {
        results_file << (*col_headers)[idc];
        results_file << " ";
    }
    results_file << endl;
  }

  results_file.close();
}




#endif