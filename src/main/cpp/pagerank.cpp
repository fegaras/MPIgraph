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

#include <set>
#include <random>
#include <cassert>

#include "graph.h"
#include "comm.h"

using namespace std;

class PageRank: public GraphPartition<long,float,char,float> {
  long num_of_vertices;
  const float damping_factor = 0.85;

public:

  PageRank ( long num_of_vertices ): GraphPartition() {
    this->num_of_vertices = num_of_vertices;
    // the zero value of merge
    zero = 0.0;
  }

  // initialize a vertex
  float initialize ( long id ) {
    return 0.05;
  }

  // merge messages
  float merge ( float x, float y ) {
    return x+y;
  }

  // calculate a new vertex value from the current vertex value
  //  and from the merged messages
  float new_value ( float val, float acc ) {
    return (1.0-damping_factor)/num_of_vertices + damping_factor*acc;
  }

  // calculate the message value to send to the edge destination;
  //   new_val is the new vertex value of the edge source;
  //   degree is the number of out-neighbors
  float send ( float new_val, char edge_val, int degree ) {
    return new_val/degree;
  }

  // if true, activate this vertex at the next superstep
  bool activate ( float new_val, float old_val ) {
    return abs(new_val-old_val)/new_val >= 0.1;
  }
};

PageRank* pagerank;

void ae ( int from, int to ) {
  if ( from % num_of_executors == executor_rank )
    pagerank->add_edge(from,to,0);
}

void test_graph () {
  // https://en.wikipedia.org/wiki/PageRank#/media/File:PageRanks-Example.svg
  if ( executor_rank == 0 )
    pagerank->add_sink(1);
  ae(2,1); ae(2,3); ae(4,2); ae(4,3); ae(4,10);
  ae(5,3); ae(5,4); ae(6,3); ae(6,4); ae(7,3); ae(7,4);
  ae(8,4); ae(9,4); ae(10,4); ae(10,3); ae(0,3); ae(3,0);
}

const int max_degree = 100;

// generates random int values according to an exponential distribution
int random_degree ( int max_degree ) {
  static default_random_engine generator(executor_rank*33);
  static exponential_distribution<double> distribution(20.0);
  double rv;
  do
    rv = distribution(generator);
  while (rv > 1.0);
  return int((max_degree-1)*rv)+1;
}

void generate_random_graph_partition ( long start, long size, long total_size ) {
  static default_random_engine gen(executor_rank*11);
  static uniform_int_distribution<long> distr(0L,total_size-1);
  set<long> s;
  for ( long i = 0; i < size; i++ )
    if (distr(gen)+1 < total_size*0.1)
      pagerank->add_sink(start+i);
    else {
      s.clear();
      for ( int j = random_degree(max_degree); j > 0; j-- )
        s.insert(distr(gen));
      for ( long e: s )
        pagerank->add_edge(start+i,e,0);
    }
}

bool cmp_values ( tuple<long,float> x, tuple<long,float> y ) {
  return get<1>(x) <= get<1>(y);
}

int main ( int argc, char* argv[] ) {
  long size = (argc > 1) ? atol(argv[1]) : 1000;
  start_comm(argc,argv);
  if (size <= 1) {
    cerr << "A graph should have more than one edge\n";
    end_comm();
    exit(-1);
  }
  int max_iterations = (argc > 2) ? atoi(argv[2]) : 1;
  pagerank = new PageRank(size);
  if (false) {
    // test a small graph
    size = 11;
    test_graph();
  } else
    generate_random_graph_partition(executor_rank*ceil(size/num_of_executors),
           (executor_rank == num_of_executors-1)
            ? size - ceil(size/num_of_executors)*(num_of_executors-1)
            : ceil(size/num_of_executors),
           size);
  // doesn't need this: pagerank->partition([&](long x)->int{ return x%num_of_executors; });
  pagerank->build_graph();
  pagerank->pregel(max_iterations);
  // collect the topk pageranks
  const int k = 10;
  set<tuple<long,float>,bool(*)(tuple<long,float>,tuple<long,float>)> topk(cmp_values);
  pagerank->foreach(
     [&](long id,float v) {
       topk.insert(tuple<long,float>(id,v));
       if (topk.size() > k)
         // remove the smallest
         topk.erase(topk.begin());
     });
  if (executor_rank == 0) {
    printf("Top-%d pageranks:\n",k);
    for ( auto it = topk.rbegin(); it != topk.rend(); it++ )
      printf("%10ld\t%0.8f\n",get<0>(*it),get<1>(*it));
  }
  end_comm();
  return 0;
}
