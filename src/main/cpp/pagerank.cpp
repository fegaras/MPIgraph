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
#include <sstream>
#include <fstream>

#include "graph.h"
#include "comm.h"

using namespace std;

class PageRank: public GraphPartition<long,float,char,float> {
  const float damping_factor = 0.85;

public:

  PageRank (): GraphPartition() {
    // the zero value of merge
    zero = 0.0;
  }

  // initialize the vertex pagerank
  inline float initialize ( long id ) const {
    return 1.0/num_of_vertices();
  }

  // merge messages (incoming contributions)
  inline float merge ( float x, float y ) const {
    return x+y;
  }

  // calculate a new vertex pagerank from the current vertex pagerank val
  //  and from the merged messages acc
  inline float new_value ( float val, float acc ) const {
    return (1.0-damping_factor)/num_of_vertices() + damping_factor*acc;
  }

  // calculate the message value to send to the edge destination;
  //   new_val is the new vertex value of the edge source;
  //   degree is the number of out-neighbors
  inline float send ( float new_val, char edge_val, int degree ) const {
    return new_val/degree;
  }

  // calculate the message value to send to all vertices (return zero to ignore)
  inline float send_all ( float new_val, int degree ) const {
    // if this is a sink, make a random jump
    return (degree == 0) ? new_val/num_of_vertices() : zero;
  }

  // if true, activate this vertex at the next superstep
  inline bool activate ( float new_val, float old_val ) const {
    return abs(new_val-old_val)/new_val >= 0.1;
  }
};

PageRank* pagerank;

void ae ( long from, long to ) {
  if ( from % num_of_executors == executor_rank )
    pagerank->add_edge(from,to,0);
}

void test_graph () {
  // simple graph from https://en.wikipedia.org/wiki/PageRank#/media/File:PageRanks-Example.svg
  if ( executor_rank == 0 )
    pagerank->add_sink(1);
  // must be ordered by source
  ae(0,3); ae(2,1); ae(2,3); ae(3,0); ae(4,2); ae(4,3); ae(4,10);
  ae(5,3); ae(5,4); ae(6,3); ae(6,4); ae(7,3); ae(7,4);
  ae(8,4); ae(9,4); ae(10,4); ae(10,3);
}

void google_web ( string file ) {
  // Google web graph: https://snap.stanford.edu/data/web-Google.html
  if ( executor_rank == 0 ) {
    ifstream infile(file);
    string line;
    while (getline(infile,line)) {
      if (line[0] == '#')
        continue;
      istringstream iss(line);
      long from, to;
      if (iss >> from >> to)
        pagerank->add_edge(from,to,0);
    }
    infile.close();
  }
}

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
  const int max_degree = 100;
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
  int max_iterations = (argc > 2) ? atoi(argv[2]) : 1;
  start_comm(argc,argv);
  pagerank = new PageRank();
  if (false) {
    // test a small graph
    test_graph();
  } else if (!isdigit(argv[1][0])) {
    // test Google web from a file
    google_web(argv[1]);
    pagerank->partition([&](long x)->int{ return x; });
  } else {
    // random graph
    long size = atol(argv[1]);
    if (size <= 1) {
      cerr << "A graph should have more than one edge\n";
      end_comm();
      exit(-1);
    }
    // sinks and edges are sorted by construction
    generate_random_graph_partition(executor_rank*ceil(size/num_of_executors),
           (executor_rank == num_of_executors-1)
            ? size - ceil(size/num_of_executors)*(num_of_executors-1)
            : ceil(size/num_of_executors),
           size);
  }
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
