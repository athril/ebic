##Using Evobic

Currently EvoBic can be used only as a command line tool.

In order to build a program simply run:
```Shell
$ make
```

Check our 'input.txt' data file to see the required input file format. In order to run an example simply type:
```Shell
$ ./evobic -i input.txt
```

The basic usage of EvoBic is:
```Shell
$ ./evobic [OPTIONS]
```

To override any of default options extra arguments should be added:

Options:
  -i,--input TEXT             input file (required!)
  -n,--iterations INT         number of iterations [default: 5000]
  -b,--biclusters INT         number of biclusters [100]
  -x,--overlap FLOAT          overlap threshold [0.75]
  -g,--gpus INT               number of gpus [1]
  -a,--approx FLOAT           approximate trends acceptance ratio [0.85]
  -m,--negative-trends INT    negative trends [1]
  -l,--log                    is logging enabled [false]


## Input
Input files are expected to have a row and a column headers, the values need to be separated by whitespace. 
The current release of EvoBic analyzes files with continuous values only. EvoBic doesn't handle missing values.
Check our toy file (input.txt) for the required file format.

## Outputs
EvoBic writes to its working directory two text files: [input_filename]-res and [input_filename]-blocks.

The first output file contains in its first line the number of detected biclusters and in consecutive lines each bicluster identifiers in the following format 
```
Bicluster([row0 row1 ... ],[column0, column1, ...])
```

The second file extracts the actual values of the bicluster from the dataset. Its format is as follows:
```
BC[bicluster_number] [number_of_rows x number_of columns]
Extracted part of the input dataset with rows and columns labels of the bicluster.
```
