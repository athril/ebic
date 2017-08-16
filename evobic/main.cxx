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


#include <iostream>
#include "CLI11.hpp"


using namespace std;

extern void start_evolution(string input_file, int MAX_ITERATIONS, int NUMBER_BICLUSTERS, float OVERLAP_THRESHOLD, float APPROX_TRENDS_RATIO, int NEGATIVE_TRENDS_ENABLED, int NUM_GPUs, bool log_enabled);


int main(int argc, char **argv) {
  CLI::App app{"EvoBic - AI-based parallel biclustering algorithm"};

  //Available options
  string input_file;
  int max_iterations=5000;
  int num_biclusters=100;
  float overlap_threshold=0.75;
  int number_of_gpus=1;
  int negative_trends_enabled=1;
  float approx_trends_ratio=0.85;
  bool log_enabled=false;


  app.add_option("-i,--input", input_file, "input file")->required();
  //app.add_option("-o,--output", output_file, "output file");
  app.add_option("-n,--iterations", max_iterations, "number of iterations [default: 5000]");
  app.add_option("-b,--biclusters", num_biclusters, "number of biclusters [100]");
  app.add_option("-x,--overlap", overlap_threshold, "overlap threshold [0.75]");
  app.add_option("-g,--gpus", number_of_gpus, "number of gpus [1]");
  app.add_option("-a,--approx", approx_trends_ratio, "approximate trends acceptance ratio [0.85]");
  app.add_option("-m,--negative-trends", negative_trends_enabled, "are negative trends enabled [1]");
  app.add_flag("-l,--log", log_enabled, "is logging enabled [false]");


  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  const static int NUM_GPUs = number_of_gpus;
  const static int MAX_ITERATIONS = max_iterations;
  const static int NUMBER_BICLUSTERS = num_biclusters;
  const static float OVERLAP_THRESHOLD = overlap_threshold;
  const static float APPROX_TRENDS_RATIO = approx_trends_ratio;
  const static float NEGATIVE_TRENDS_ENABLED = negative_trends_enabled;

  cout << "Running EvoBic with following options: ./evobic -i " << input_file << " -n " << MAX_ITERATIONS << " -b " << NUMBER_BICLUSTERS << " -x " << OVERLAP_THRESHOLD << " -a " << APPROX_TRENDS_RATIO << " -m " << NEGATIVE_TRENDS_ENABLED << endl;

  //srand(time(NULL));
  srand(1);
  start_evolution(input_file, MAX_ITERATIONS, NUMBER_BICLUSTERS, OVERLAP_THRESHOLD, APPROX_TRENDS_RATIO, NEGATIVE_TRENDS_ENABLED, NUM_GPUs, log_enabled);


  return 0;
}
