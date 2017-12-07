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


#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

const static int TOURNAMENT_SIZE = 50;
//const static int MAX_NUMBER_OF_TABU_HITS = POPULATION_SIZE;

const static float RATE_CROSSOVER = 0.2;
const static float RATE_MUTATION = 0.8;
const static float RATE_MUTATION_SWAP = 0.2;
const static float RATE_MUTATION_SUBSTITUTION = 0.4+RATE_MUTATION_SWAP;
const static float RATE_MUTATION_INSERTION = 0.3+RATE_MUTATION_SUBSTITUTION;
const static float RATE_MUTATION_DELETION = 0.1+RATE_MUTATION_INSERTION;

const static int MIN_COLS_CHROMOSOME = 2;
const static int MAX_COLS_CHROMOSOME = 4;

const static float OVERLAP_PENALTY = 1.2;

const static int MIN_NO_COLS = 4;
const static int MIN_NO_ROWS =10;

const static float EPSILON=0.000001;

#define MAX(a,b) ( ((a) > (b)) ? (a) : (b) )
#define MIN(a,b) ( ((a) < (b)) ? (a) : (b) )

#endif