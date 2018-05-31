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
#include <vector>
#include <algorithm>
#include <cfloat>
#include <iterator>
#include <set>
#include <assert.h>
#include <libgen.h>
#include <unordered_set>
#include <string.h>
#include "parameters.hxx"
#include "ebic.hxx"

#define MAX(a,b) ( ((a) > (b)) ? (a) : (b) )

typedef std::vector<int> chromosome;
typedef std::vector< std::vector<int> > population;
typedef std::ostringstream logger;
typedef population::iterator myiter;


std::ostream& operator<<(std::ostream& os, const chromosome& c) {
  for (auto x : c) 
    os << x << " ";
  return os;
}

std::ostream& operator<<(std::ostream& os, const vector<chromosome>& c) {
  for (auto x : c) 
    os << "["<< x << "]";
  return os;
}



float find_intersection(chromosome a, chromosome b) {
  chromosome intersection;
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(intersection));
  return MAX( (float)intersection.size()/(float)a.size(), (float)intersection.size()/(float)b.size() );
}


bool crossover(chromosome &parent_a, chromosome &parent_b, chromosome &offspring_a, logger &l) { 
  auto cut_a=rand() % (parent_a.size()-1)+1;  // 0...size-1
  auto cut_b=rand() % (parent_b.size()-1);  // 0...size-1/*

  //log << cut_a << " " << cut_b << endl;
  //log << "CROSSOVER: [" << parent_a << " (" <<cut_a << ")" << " x " << parent_b;
  std::copy_n(parent_a.begin(), cut_a, std::back_inserter(offspring_a));
  //std::copy_n(parent_b.begin(), cut_b, std::back_inserter(offspring_b));
#ifdef DEBUG
  l << "[" << parent_a << "x " << parent_b << "]-X->";
#endif
  //log << "START:" << offspring_a << endl;

  for (auto it=parent_b.begin()+cut_b; it!=parent_b.end(); ++it) {
    if (std::find(offspring_a.begin(), offspring_a.end(), *it) == offspring_a.end()) {
      offspring_a.push_back(*it);
      //log << " ->" << offspring_a << endl;
    }
  }
  l << offspring_a;
  return true;
}

bool mutation_substitution( chromosome &chromo, int num_columns, logger &l) {
  // log << "MUTATION_SUB: " << chromo;
  auto mutation_point = rand() % chromo.size();
  auto mutation_value = rand() % num_columns;
  if (std::find(chromo.begin(), chromo.end(), mutation_value) != chromo.end())
    return false;
#ifdef DEBUG
  l << "[" << chromo << "]" << "-@->";
#endif
  chromo.at(mutation_point)=mutation_value;
  //log << "->" << chromo << endl;  
  l << "[" << chromo << "]";
  return true;
}

bool mutation_insertion( chromosome &chromo, int num_columns, logger &l ) {
  // log << "MUTATION_INS: " << chromo;
  auto mutation_point = rand() % chromo.size();
  auto mutation_value = rand() % num_columns;
  if (std::find(chromo.begin(), chromo.end(), mutation_value) != chromo.end())
    return false;
#ifdef DEBUG
  l << "[" << chromo << "]" << "-I->";
#endif
  chromo.insert(chromo.begin()+mutation_point,mutation_value);
  //log << "->" << chromo << endl;
  l << "[" << chromo << "]";
  return true;
}


bool mutation_deletion( chromosome &chromo, logger &l ) {
  if (chromo.size()<=2)
    return false;
  // log << "MUTATION_DEL: " << chromo;
#ifdef DEBUG
  l << "[" << chromo << "]" << "-D->";  
#endif
  auto mutation_point = rand() % chromo.size();
  chromo.erase(chromo.begin()+mutation_point);
  //log << "->" << chromo << endl;
#ifdef DEBUG
  l << "[" << chromo << "]";
#endif
  return true;
}


bool mutation_swap( chromosome &chromo, logger &l ) {
  // log << "MUTATION_SWP: " << chromo;
#ifdef DEBUG
  l << "[" << chromo << "]" << "-S->";
#endif
  auto mutation_point1 = rand() % chromo.size();
  auto mutation_point2 = rand() % chromo.size();
  if (mutation_point1==mutation_point2)
    return false;
  auto tmp=chromo.at(mutation_point1);
  chromo.at(mutation_point1)=chromo.at(mutation_point2);
  chromo.at(mutation_point2)=tmp;
  //log << "->" << chromo << endl;
#ifdef DEBUG
  l << "[" << chromo << "]";
#endif
  return true;
}


chromosome& generate_chromosome (chromosome& chromo, int MIN_SIZE, int MAX_SIZE, int NUM_COLUMNS ) {
  auto size = rand() % (MAX_SIZE - MIN_SIZE) + MIN_SIZE;    
  auto column=rand() % NUM_COLUMNS;
  for (auto i=0; i<size; ++i) {
    while (std::find(chromo.begin(), chromo.end(), column) != chromo.end()) {
      column=rand() % NUM_COLUMNS;
    }
    chromo.push_back(column);
  }
  return chromo;
}


int array_inserter(population &population, int *ruleset_indices, int *compressed_ruleset, int is_forward=1) {
  size_t total_size=0;
  for (auto i=0; i<population.size(); i++) {
    auto pos=0;
    ruleset_indices[i]=total_size;
    for (auto v : population.at(i)) {
      compressed_ruleset[total_size+(is_forward ? population.at(i).size()-pos-1 : pos)]=v;
      pos++;
    }
    total_size+=population.at(i).size();
  }
  ruleset_indices[population.size()]=total_size;
  return total_size;
}

int get_compressed_size(population &population) {
  auto total_size=0;
  for (auto i=0; i<population.size(); i++) {
      total_size+=population.at(i).size();
  }
  return total_size;
}



bool tabu_approved_addition(population &population, chromosome &chromo, std::unordered_set<string> &tabu, int *penalties ) {
  if (chromo.size()<2 )
    return false;
  std::ostringstream stream;
  //stream << *chromo.begin() << "|" << *chromo.rbegin();
  stream << chromo;
  if ( tabu.find(stream.str()) != tabu.end() )
    return false;
  for (auto node : chromo) {
    penalties[node]++;            
  }

  tabu.insert(stream.str());
  population.push_back(chromo);
  return true;
}

//returns ID of selected chromosome
int tournament_selection(float *fitness, int *penalties, int tournament_size, population &population, int population_size, int num_biclusters) {
  auto bestId=-1;
  auto best_fitness=-1.0f;
  for (auto i=0; i<tournament_size; ++i) {
    auto id= rand() % population.size();
    float total_penalty=0;
    
    for (auto val : population.at(id)) {
      total_penalty+=penalties[val];
    }
    total_penalty/=population.at(id).size();
    total_penalty=pow(OVERLAP_PENALTY,total_penalty);
    if (MAX(0,fitness[id]-total_penalty) >= best_fitness) {
      best_fitness = fitness[id];
      bestId=id;
    }
  }
  return bestId;
}





void perform_evolutions(EBic &ebic, string input_file, int MAX_ITERATIONS, int MAX_NUMBER_BICLUSTERS, int num_columns, float OVERLAP_THRESHOLD, int POPULATION_SIZE, bool log_enabled) {
  const static int REPRODUCTION_SIZE = POPULATION_SIZE/4;
  const static int MAX_NUMBER_OF_TABU_HITS = POPULATION_SIZE;

  logger log;

  std::set< std::pair<float, std::vector<int> > > toplist;

  population population_old, population_new;

  std::unordered_set<string> tabu_list;
  int *penalties = new int[num_columns];

  for (int i=0; i<num_columns; ++i)
    penalties[i]=1;

  //for (auto counter=0; counter<POPULATION_SIZE;) {
  while(population_old.size()<POPULATION_SIZE) {
    chromosome c;
    generate_chromosome(c,MIN_COLS_CHROMOSOME,MAX_COLS_CHROMOSOME,num_columns);
    tabu_approved_addition(population_old, c, tabu_list, penalties);
  }
  
  float *fitness=new float[POPULATION_SIZE];
  float *prev_fitness=new float[POPULATION_SIZE];
  int *rules_indices=new int[POPULATION_SIZE+1];
  int *compressed_ruleset=new int[get_compressed_size(population_old)];



  
  array_inserter(population_old, rules_indices, compressed_ruleset);
  //problem_t problem = {POPULATION_SIZE, rules_indices, get_compressed_size(population_old), compressed_ruleset, NUM_ROWS, NUM_COLUMNS, data, row_headers, col_headers};
  problem_t problem = {POPULATION_SIZE, rules_indices, get_compressed_size(population_old), compressed_ruleset};
  ebic.determine_fitness(&problem, fitness);
  //log << "GPU-->--" << endl;
  for (int i=0; i<POPULATION_SIZE; i++)
    prev_fitness[i]=fitness[i];
  
  //logger *log = new logger[POPULATION_SIZE];
  assert(REPRODUCTION_SIZE>1);


  //toplist preparation
  for (int i=0; i<POPULATION_SIZE && toplist.size()<REPRODUCTION_SIZE; ++i) {    
    bool addition_allowed=true;
    for (auto t=toplist.rbegin(); t!=toplist.rend(); ++t) {
      if (fitness[i] < t->first) {
        if (find_intersection(population_old.at(i), t->second)>=OVERLAP_THRESHOLD) {
          addition_allowed=false;
          break;
        }
      }
      //addition allowed
      if (find_intersection(population_old.at(i), t->second)>=OVERLAP_THRESHOLD) {
        toplist.erase(--t.base());
      }
    }
    if (addition_allowed && population_old.at(i).size()>=MIN_NO_COLS && fitness[i]>0) {
      toplist.insert(make_pair(fitness[i],population_old.at(i)));
    }
  }

  //GP - ALGORITHM ITERATIONS START HERE................
  for (int iteration=0; iteration<MAX_ITERATIONS; ++iteration) {    
    for (auto t=toplist.rbegin(); t!=toplist.rend() && population_new.size()<=REPRODUCTION_SIZE; ++t) {
      population_new.push_back(t->second);
    }
    int tabu_hits=0;
    logger tmp_log;
    while ( population_new.size() < POPULATION_SIZE) {
      int id=tournament_selection( fitness, penalties, TOURNAMENT_SIZE, population_old, POPULATION_SIZE, MAX_NUMBER_BICLUSTERS);
      chromosome chromo = population_old.at( id );
      bool added=true;
#ifdef DEBUG
      tmp_log << " (" << fitness[id] << ")";
#endif
      if ( (float)rand()/(float)RAND_MAX < RATE_CROSSOVER) {
        int id2=tournament_selection( fitness, penalties, TOURNAMENT_SIZE, population_old, POPULATION_SIZE, MAX_NUMBER_BICLUSTERS);
        tmp_log << " (" << fitness[id2] << ")";
        chromosome chromo_b =population_old.at( id2 );
        chromosome offspring;
        added&=crossover(  chromo, chromo_b, offspring, tmp_log);
        chromo=offspring;
      }
      else {
        if ( (float)rand()/(float)RAND_MAX < RATE_MUTATION ) {
          float mutation_type=(float)rand()/(float)RAND_MAX; //0..1
          if ( mutation_type < RATE_MUTATION_SWAP ) {
            added&=mutation_swap(chromo, tmp_log);
          }
          else if ( mutation_type-RATE_MUTATION_SWAP < RATE_MUTATION_SUBSTITUTION ) {
            added&=mutation_substitution(chromo, num_columns, tmp_log);
          }
          else if ( mutation_type-RATE_MUTATION_SWAP-RATE_MUTATION_SUBSTITUTION < RATE_MUTATION_INSERTION){
            added&=mutation_insertion(chromo, num_columns, tmp_log);
          }
          else if ( mutation_type-RATE_MUTATION_SWAP-RATE_MUTATION_SUBSTITUTION-RATE_MUTATION_INSERTION < RATE_MUTATION_DELETION) {
            added&=mutation_deletion(chromo, tmp_log);
          }          
        } //end mutation
      } //end else
      if(added) {
        tabu_approved_addition(population_new, chromo, tabu_list, penalties);
      } else {
        tabu_hits++;
#ifdef DEBUG
        log << "X";
#endif

      }



#ifdef DEBUG
      tmp_log.str("");
      tmp_log.clear();
#endif

    } // end while (population per bicluster)

#ifdef DEBUG
    log << endl;
#endif

    for (int i=0; i<POPULATION_SIZE; i++)  {
      prev_fitness[i]=fitness[i];
    }

    int size_indices = get_compressed_size(population_new);
    int *compressed_ruleset=new int[size_indices];

    array_inserter(population_new, rules_indices, compressed_ruleset);
    problem_t problem = {POPULATION_SIZE, rules_indices, size_indices, compressed_ruleset};//, NUM_ROWS, NUM_COLUMNS, data, row_headers, col_headers};

    ebic.determine_fitness(&problem, fitness);
    //log << "GPU-->--" << endl;

    std::vector< std::pair<float, myiter> > order( population_new.size() );  
    std::vector< std::pair<float, int> > order_indices( population_new.size() );  
    size_t n = 0;
    
    for (myiter it = population_new.begin(); it != population_new.end(); ++it, n++) {
      int total_penalty=0;
      order[n] = make_pair(fitness[n]-total_penalty, it);
      order_indices[n] = make_pair(fitness[n]-total_penalty, n);
    }
    std::sort(order.begin(), order.end(), [](std::pair<float, myiter> const& a, std::pair<float, myiter> const& b) {return a.first > b.first;});
    std::sort(order_indices.begin(), order_indices.end(), [](std::pair<float, int> const& a, std::pair<float, int> const& b) {return a.first > b.first;});

    for (int i=0; i<POPULATION_SIZE; ++i) {    
      bool addition_allowed=true;
      for (auto t=toplist.rbegin(); t!=toplist.rend();) {
        if (fitness[order_indices.at(i).second] < t->first) { 
          if (find_intersection(*order.at(i).second, t->second)>OVERLAP_THRESHOLD || fitness[order_indices.at(i).second]==0) {
            addition_allowed=false;
            i=POPULATION_SIZE;
            break;
          }
        }
        //addition allowed
        if (find_intersection(*order.at(i).second, t->second)>OVERLAP_THRESHOLD) {
            toplist.erase(--t.base());
        } else {
          t++;
        }
      }
      if (addition_allowed && order.at(i).second->size()>=MIN_NO_COLS && fitness[order_indices.at(i).second]>0) {
        toplist.insert(make_pair(fitness[order_indices.at(i).second],*order.at(i).second));
      }
    }    

    for (auto t=toplist.begin(); t!=toplist.end() && toplist.size()>REPRODUCTION_SIZE; ++t) {
      toplist.erase(t);
    }
#ifdef DEBUG
    log << "TOPLIST " << "[" << toplist.size() << "] - iteration: "<< iteration << endl;
#endif
    int counter=0;
    for (auto t=toplist.rbegin(); t!=toplist.rend(); ++t, ++counter) {
    //for (auto t : toplist) {
#ifdef DEBUG
        log << t->first << "->" << t->second << " ("<< t->second.size() << ")"<< endl;
#endif

        if (counter>MAX_NUMBER_BICLUSTERS)
            break;
    }

#ifdef DEBUG
    log << endl;
#endif

    population_old.clear();
    population_old=population_new;
    population_new.clear();
//    if (!(iteration % 1000)) {
      cout << "Iteration: " << iteration << endl;
/*
#ifdef DEBUG_MODE
      for (auto t=toplist.rbegin(); t!=toplist.rend(); ++t) {
        population_new.push_back(t->second);
      }
      int size_indices = get_compressed_size(population_new);
      compressed_ruleset=new int[size_indices];
      array_inserter(population_new, rules_indices, compressed_ruleset);
      problem = {MIN((int)population_new.size(), MAX_NUMBER_BICLUSTERS), rules_indices, size_indices, compressed_ruleset, NUM_ROWS, NUM_COLUMNS, data, row_headers, col_headers};
      stringstream temp_log;
      temp_log << result_filename.str() << "-" << iteration;
      ebic.get_final_biclusters(&problem);
      ebic.print_biclusters_synthformat(&problem, temp_log.str());
    }
#endif
*/  
//    }
    population_new.clear();
#ifdef DEBUG
    log << "TABU list size:" <<tabu_list.size() << endl; 
#endif
    if (tabu_hits>=MAX_NUMBER_OF_TABU_HITS)
      break;
    for (int i=0; i<num_columns; i++) {
#ifdef DEBUG
        log << penalties[i] << " ";
#endif
        penalties[i]=0;
    }

#ifdef DEBUG
    log << endl;
    log << "--------------------------------"<<endl;
#endif

 }
#ifdef DEBUG
  log << "TOPLIST - final biclusters: " << endl;
  for (auto topbi : toplist) {
    log << topbi.first << "->" << topbi.second << " ("<< topbi.second.size() << ")"<< endl;
  }
#endif
  for (auto t=toplist.rbegin(); t!=toplist.rend(); ++t) {
    population_new.push_back(t->second);
    if (population_new.size()>=MAX_NUMBER_BICLUSTERS)
      break;
  }
  
  int size_indices = get_compressed_size(population_new);
  compressed_ruleset=new int[size_indices];
  array_inserter(population_new, rules_indices, compressed_ruleset);

  problem = {MIN((int)population_new.size(), MAX_NUMBER_BICLUSTERS), rules_indices, size_indices, compressed_ruleset};//, NUM_ROWS, NUM_COLUMNS, data, row_headers, col_headers};
  ebic.get_final_biclusters(&problem);

#ifdef DEBUG
  if (log_enabled) {
    std::stringstream log_name;
    log_name << basename(strdup(input_file.c_str())) <<"-log";
    std::ofstream log_file;
    log_file.open(log_name.str().c_str());
    log_file << log.str();
    log_file.close();
  }
#endif
  cout << "Results were written to '" << input_file << "-blocks' and '" << input_file << "-res' files." << endl;

  return;
}
