/*
 * Copyright © 2025 Leonidas Fegaras, University of Texas at Arlington
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <functional>
#include <set>
#include <random>
#include <iostream>
#include <cassert>
#include "omp.h"

#include "graph.h"
#include "comm.h"

using namespace std;

class PageRank: public GraphPartition<int,float,char,float> {
public:
  PageRank (): GraphPartition() {
    max_iterations = 10;
  }

  float initialize ( int id ) {
    return 0.1;
  }

  float zero = 0.0;

  float merge ( float x, float y ) {
    return x+y;
  }

  float new_value ( float val, float acc ) {
    return acc;
  }

  float send ( float new_val, char edge_val, int degree ) {
    return new_val/degree;
  }

  bool activate ( float new_val, float old_val ) {
    return true;
  }
};

PageRank* pagerank;

const int max_degree = 100;

// generates random int values according to an exponential distribution
int random_degree ( int max_degree ) {
  static default_random_engine generator(executor_rank*33);
  static exponential_distribution<double> distribution(10.0);
  double rv;
  do
    rv = distribution(generator);
  while (rv > 1.0);
  return int((max_degree-1)*rv)+1;
}

void generate_random_graph_partition ( int start, int size, int total_size ) {
  static default_random_engine gen(executor_rank*11);
  static uniform_int_distribution<int> distr(0,total_size-1);
  set<int> s;
  for ( int i = 0; i < size; i++ )
    if (distr(gen) < ceil(total_size/30))
      pagerank->add_sink(start+i);
    else {
      s.clear();
      for ( int j = random_degree(max_degree); j > 0; j-- )
        s.insert(distr(gen));
      for ( int e: s )
        pagerank->add_edge(start+i,e,0);
    }
}

bool cmp_values ( tuple<int,float> x, tuple<int,float> y ) {
  return get<1>(x) >= get<1>(y);
}

int main ( int argc, char* argv[] ) {
  // vertices per partition
  int size = (argc > 1) ? atoi(argv[1]) : 1000;
  start_comm(argc,argv);
  assert(size > 1);
  pagerank = new PageRank();
  generate_random_graph_partition(executor_rank*ceil(size/num_of_executors),
                                  (executor_rank == num_of_executors-1)
                                  ? size - ceil(size/num_of_executors)*(num_of_executors-1)
                                  : ceil(size/num_of_executors),
                                  size);
  pagerank->build_graph();
  if (argc > 2)
    pagerank->max_iterations = atoi(argv[2]);
  pagerank->pregel();
  // collect and print the topk pageranks
  const int k = 10;
  set<tuple<int,float>,bool(*)(tuple<int,float>,tuple<int,float>)> topk (cmp_values);
  pagerank->collect(
     [&](int id,float v) {
       auto t = tuple<int,float>(id,v);
       topk.insert(t);
       if (topk.size() == k) {
         auto it = topk.begin();
         for ( int i = 1; i < k; i++ )
           it++;
         topk.erase(it);
       }
     });
  if (executor_rank == 0) {
    printf("Top-%d pageranks:\n",k);
    for ( auto t: topk )
      printf("%10d\t%0.3f\n",get<0>(t),get<1>(t));
  }
  end_comm();
  return 0;
}
