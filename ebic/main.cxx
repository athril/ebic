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


#include <iostream>
#include <sstream>
#include <cfloat>
#include <assert.h>
#include <string.h>
#include "CLI11.hpp"
#include "dataIO.hxx"
#include "ebic.hxx"
#include <chrono>


using namespace std;

extern void perform_evolutions(EBic &ebic, string input_file, int MAX_ITERATIONS, int NUMBER_BICLUSTERS, int num_columns, float OVERLAP_THRESHOLD, int MAX_NUMBER_OF_TABU_HITS, bool log_enabled);


int main(int argc, char **argv) {
  CLI::App app{"ebic - AI-based parallel biclustering algorithm"};

  //Available options
  string input_file;
  int max_iterations=5000;
  int num_biclusters=100;
  float overlap_threshold=0.75;
  int number_of_gpus=1;
  int negative_trends_enabled=1;
  float approx_trends_ratio=0.85;
  bool log_enabled=false;
  float missing_value=FLT_MAX;

  app.add_option("-i,--input", input_file, "input file")->required();
  //app.add_option("-o,--output", output_file, "output file");
  app.add_option("-n,--iterations", max_iterations, "number of iterations [default: 5000]");
  app.add_option("-b,--biclusters", num_biclusters, "number of biclusters [100]");
  app.add_option("-x,--overlap", overlap_threshold, "overlap threshold [0.75]");
  app.add_option("-g,--gpus", number_of_gpus, "number of gpus [1]");
  app.add_option("-a,--approx", approx_trends_ratio, "approximate trends allowance [0.85]");
  app.add_option("-t,--negative-trends", negative_trends_enabled, "are negative trends enabled [1]");
  app.add_option("-m,--missing_value", missing_value, "numeric representation of missing value [MAX]");

  app.add_flag("-l,--log", log_enabled, "is logging enabled [false]");


  try {
    app.parse(argc, argv);
    std::stringstream result_filename,result_filename2,result_filename3;
    result_filename << strdup(input_file.c_str()) <<"-res.txt";
    result_filename2 << strdup(input_file.c_str()) <<"-blocks.txt";
    result_filename3 << strdup(input_file.c_str()) <<"-r.txt";

    //setting all parameters
    int num_rows, num_columns;
    vector<float> input_data;
    vector<string> row_headers, col_headers;

    //loading dataset and assessing its size
    load_data(input_file, input_data, num_rows, num_columns, row_headers, col_headers);
    if (num_columns<=20) 
      cerr << "In order to exploit the full potential of EBIC the dataset needs to have at least 20 columns." << endl;

    //srand(time(NULL));
    srand(1);

  int trends_population_size = MIN(pow(2,num_columns-1),1600);
//  int trends_population_size = 65535; // max number allowed
//    int trends_population_size = 10000;


    float *data = &input_data[0];

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    EBic ebic(number_of_gpus, approx_trends_ratio, negative_trends_enabled, trends_population_size, missing_value, data, num_rows, num_columns, row_headers, col_headers);

    cout << "Running ebic with the following parameters: ./ebic -i " << input_file << " -n " << max_iterations << " -b " << num_biclusters << " -x " << overlap_threshold << " -a " << approx_trends_ratio << " -t " << negative_trends_enabled << endl;

    perform_evolutions(ebic, input_file, max_iterations, num_biclusters, num_columns, overlap_threshold, trends_population_size, log_enabled);

    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " microseconds" << std::endl;

    ebic.print_biclusters_synthformat(result_filename.str());
    ebic.print_biclusters_blocks(result_filename2.str());
    ebic.print_biclusters_r(result_filename3.str());
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }


  return 0;
}
