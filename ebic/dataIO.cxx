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


#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include "dataIO.hxx"


void load_data(string dataset_filename, vector<float> &input_data, int &num_rows, int &num_cols, vector<string> &row_headers, vector<string> &col_headers) {
  std::ifstream datafile(dataset_filename.c_str());
  if (!datafile.good())
    exit(1);
  std::string line;
  int rowid=0;

  std::getline(datafile,line);
  istringstream iss(line);

  std::vector<string> tokens{istream_iterator<string>{iss}, istream_iterator<string>{}};
  //check if the first header is empty
  if (line.find_first_not_of("\t ")==0)
    tokens.erase(tokens.begin());

  num_cols=tokens.size();
  copy(tokens.begin(),tokens.end(),back_inserter(col_headers));

  num_rows=0;

  while (std::getline(datafile, line) ) {
    istringstream iss(line);
    std::vector<string> tokens{istream_iterator<string>{iss}, istream_iterator<string>{}};
    row_headers.push_back(tokens[0]);
    tokens.erase(tokens.begin());

    std::transform(std::begin(tokens), std::end(tokens), std::back_inserter(input_data), [](string s) { return std::stof(s); } );
    num_rows++;
  }
  //std::cout << "PROBLEM SIZE: " << num_rows << " " << num_cols << std::endl;
  datafile.close();
}
