## EBIC - AI-based parallel biclustering algorithm

*EBIC* is a next-generation biclustering algorithm based on artificial intelligence (AI). 
*EBIC* is probably the first algorithm capable of discovering the most challenging patterns (i.e. row-constant, column-constant, shift, scale, shift-scale and trend-preserving) in complex and noisy data with average accuracy of over 90%.
It is also one of the very few parallel biclustering algorithms that use at least one graphics processing unit (GPU) and is ready for big-data challenges.

*EBIC* is mainly implemented in C++11. CUDA with OpenMP used for parallelization.

**EBIC is still under active development**


## License

*EBIC* is MIT-licensed. Please see the [repository license](https://github.com/athril/ebic/blob/master/LICENSE) for the licensing and usage information.


## Installation

*EBIC* requires CUDA 8.0, installed C++11 environment and OpenMP.
We maintain the [EBIC installation instructions](http://athril.github.io/ebic/installation/).


## Usage

*EBIC* can be used only as a [command line tool](http://athril.github.io/ebic/usage/).

In order to build a program simply run:
```$ make```

Check our 'input.txt' data file to see the required input file format. In order to run an example simply type:
```$ ./ebic -i input.txt```

The basic usage of EBIC is: 
```$ ./ebic [OPTIONS]```

To override any of default options extra arguments should be added:

Options:

*  -i,--input TEXT             input file 

*  -n,--iterations INT         number of iterations [default: 5000]

*  -b,--biclusters INT         number of biclusters [100]

*  -x,--overlap FLOAT          overlap threshold [0.75]

*  -g,--gpus INT               number of requested GPUs [1]

*  -a,--approx FLOAT           approximate trends allowance [0.85]

*  -t,--negative-trends INT    negative trends [1]

*  -l,--log                    is logging enabled [false]



## Examples

Check available options:
```Shell
$ ./ebic -h
```

Run EBIC for 10 iterations and return 5 biclusters only:
```Shell
$ ./ebic -i input.txt -n 10 -b 5
```

Do not allow negative trends:
```Shell
$ ./ebic -i input.txt -t 0
```

Do not allow approximate trends:
```Shell
$ ./ebic -i input.txt -a 1
```

