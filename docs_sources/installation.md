## EvoBic prerequisites

EvoBic requires CUDA Toolkit (7.0 or later) and C++11 environment to be installed. Please make sure that your computer may run CUDA.


Find out your graphic card and check if it is supported on [NVIDIA website](https://developer.nvidia.com/cuda-gpus).

You won't be able to run EvoBic if your graphic card is not supported.



### NVIDIA driver installation

Start with installing NVIDIA driver for your graphic card.

Find the latest NVIDIA driver at [NVIDIA's Download Website](http://www.nvidia.com/Download/index.aspx?lang=en-us)



### CUDA Toolkit installation

Follow NVIDIA tutorial for [CUDA](https://developer.nvidia.com/cuda-downloads). Install the latest supported CUDA Toolkit for your platform.

## GCC 4.6+ installation
Windows users may also require to install a GCC 4.6+ or newer compiler in order to run C++11 code.
The easiest way is downloading the [MinGW-w64](http://mingw-w64.org/doku.php/download#mingw-builds).



## EvoBic compilation

In order to prepare a binary type:
```Shell
make
```
or
```Shell
nvcc -O3 -std=c++11 --expt-extended-lambda -Xcompiler -fopenmp --default-stream per-thread  main.cxx population.cxx dataIO.cxx evobic.cu -L/usr/local/cuda/lib -lcuda -o evobic
```
If the command completed successfully, you are ready to go!
