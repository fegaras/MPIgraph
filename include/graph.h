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
#include <vector>
#include <tuple>
#include <climits>
#include <algorithm>
#include <thread>
#include "mpi.h"

#include "serialization.h"
#include "comm.h"

using namespace std;

const bool trace = false;

/* A graph partition of an executor. A partition is associated with a set of vertices 
   that do not overlap with the vertices in other partitions. It also contains all
   edges whose source is one of these vertices.
   ID: the type of a vertex id
   V: the type of a vertex value
   E: the type of an edge value
   M: message type
*/
template< typename ID, typename V, typename E, typename M >
class GraphPartition {
private:

  // total number of partitions (= num_of_executors)
  int num_of_partitions;

  // index of the current partition in partitions (= executor_rank)
  int current_partition;

  // each partition is associated with the index of its first vertex
  vector<int> partitions;

  int total_num_of_vertices;

  typedef struct {
    ID id;           // vertex id
    V value;         // vertex value
    M message;       // the incoming message to this vertex
    int first_edge;  // index of the first edge of this vertex (in edges)
    bool active;     // is this vertex active?
  } vertex_t;
  // local vertices
  vector<vertex_t> vertices;

  typedef struct {
    int destination; // the index of the edge destination
    E value;         // edge value
    M message;       // the outgoing message to be send to the edge destination
  } edge_t;
  // contains the out-neighbors of the local vertices
  vector<edge_t> edges;

  // temporary storage of all vertices (one vector<ID> for each partition)
  vector<vector<ID>>* tmp_vertices;
  // a sink is a vertex that doesn't have any out-neighbors
  vector<ID>* tmp_sinks;
  // temporary storage of local edges
  vector<tuple<ID,ID,E>>* tmp_edges;

  // get the index of a vertex id
  int id_loc ( ID id );

public:

  int max_iterations = INT_MAX;

  GraphPartition () {
    num_of_partitions = num_of_executors;
    current_partition = executor_rank;
    partitions = vector<int>(num_of_partitions,0);
    tmp_vertices = new vector<vector<ID>>();
    tmp_sinks = new vector<ID>();
    for ( int i = 0; i < num_of_partitions; i++ )
      tmp_vertices->push_back(vector<ID>());
    tmp_edges = new vector<tuple<ID,ID,E>>();
  }

  // add a graph edge
  void add_edge ( ID from, ID to, E edge_val ) {
    if (trace)
      printf("adding edge %d -> %d\n",from,to);
    tmp_edges->push_back(tuple<ID,ID,E>(from,to,edge_val));
  }

  // add a sink (a vertex that doesn't have any out-neighbors)
  void add_sink ( ID vertex ) {
    if (trace)
      printf("adding sink %d\n",vertex);
    tmp_sinks->push_back(vertex);
  }

  // build the graph partitions from the edges and the sinks
  void build_graph ();

  // initialize a vertex
  virtual V initialize ( ID id ) = 0;

  // the zero value of merge
  M zero;

  // merge messages
  virtual M merge ( M x, M y ) = 0;

  // calculate a new vertex value from the current vertex value
  //  and from the merged messages
  virtual V new_value ( V val, M acc ) = 0;

  // calculate the message value to send to the edge destination;
  //   new_val is the new vertex value of the edge source;
  //   degree is the number of out-neighbors
  virtual M send ( V new_val, E edge_val, int degree ) = 0;

  // if true, activate this vertex in the next superstep
  virtual bool activate ( V old_val, V new_val ) = 0;

  // Pregel graph processing
  void pregel ( int max_iterations );

  // brink the vertices at the coordinator and apply f to each vertex 
  void collect ( function<void(ID,V)> f );
};

// get the index of a vertex id
template< typename ID, typename V, typename E, typename M >
int GraphPartition<ID,V,E,M>::id_loc ( ID id ) {
  for ( int i = 0; i < num_of_partitions; i++ ) {
    auto first = (*tmp_vertices)[i].begin();
    auto last = (*tmp_vertices)[i].end();
    auto it = lower_bound(first,last,id);
    if (it != last && id >= *it)
      return (it-first)+partitions[i];
  }
  return 0;
}

// build the graph partition from the edges and the sinks
template< typename ID, typename V, typename E, typename M >
void GraphPartition<ID,V,E,M>::build_graph () {
  // build the local graph partition from the tmp vectors
  sort(tmp_edges->begin(),tmp_edges->end(),
       [&](tuple<ID,ID,E> x,tuple<ID,ID,E> y)->bool {
           return get<0>(x) < get<0>(y);
       });
  sort(tmp_sinks->begin(),tmp_sinks->end());
  ID vid;
  int index = 0;
  auto sink_it = tmp_sinks->begin();
  for ( auto &t: *tmp_edges ) {
    while (sink_it != tmp_sinks->end() && *sink_it < get<0>(t)) {
      // store the sinks in vertices
      vertex_t v;
      v.id = *sink_it;
      v.first_edge = index;
      vertices.push_back(v);
      (*tmp_vertices)[current_partition].push_back(v.id);
      sink_it++;
    }
    if (index == 0 || get<0>(t) != vid) {
      // store a unique edge source in vertices
      vid = get<0>(t);
      vertex_t v;
      v.id = vid;
      v.first_edge = index;
      vertices.push_back(v);
      (*tmp_vertices)[current_partition].push_back(vid);
    } // else skip this edge since its source is already stored
    index++;
  }
  while (sink_it != tmp_sinks->end()) {
    vertex_t v;
    v.id = *sink_it;
    v.first_edge = index-1;
    vertices.push_back(v);
    (*tmp_vertices)[current_partition].push_back(v.id);
    sink_it++;
  }
  delete tmp_sinks;
  // create a thread to read the local vertices from all other ranks
  thread trcv(
      [&]()->void {
        int buffer_size = sizeof(size_t)
              +((*tmp_vertices)[current_partition].size()*num_of_partitions)*sizeof(ID);
        char* buffer = new char[buffer_size];
        for ( int i = 1; i < num_of_partitions; i++ ) {
          int rank = receive_data(buffer,buffer_size);
          deserialize((*tmp_vertices)[rank],buffer,buffer_size);
        }
        delete[] buffer;
      });
  char* buffer;
  int buffer_size = serialize((*tmp_vertices)[current_partition],buffer);
  // broadcast the local vertices to all other ranks
  for ( int i = 0; i < num_of_partitions; i++ )
    if (i != current_partition)
      send_data(i,buffer,buffer_size,0);
  trcv.join();
  delete[] buffer;
  // build the partition table
  int n = 0;
  int i = 0;
  for ( auto tv: *tmp_vertices ) {
    partitions[i++] = n;
    n += tv.size();
  }
  total_num_of_vertices = n;
  // store edges using indices (offsets) for destination
  for ( auto &te: *tmp_edges ) {
    edge_t e;
    e.destination = id_loc(get<1>(te));
    e.value = get<2>(te);
    edges.push_back(e);
  }
  float size = (vertices.size()*sizeof(vertex_t)
                +edges.size()*sizeof(edge_t))/1024.0/1024.0;
  printf("Executor %d: vertices=%ld, edges=%ld, cache=%.3f MBs\n",
         executor_rank,vertices.size(),edges.size(),size);
  delete tmp_vertices;
  delete tmp_edges;
  barrier();
  i = 0;
} 

// Pregel graph processing
template< typename ID, typename V, typename E, typename M >
void GraphPartition<ID,V,E,M>::pregel ( int max_iterations ) {
  vector<tuple<int,M>> outgoing;
  vector<tuple<int,M>> out_msgs;
  this->max_iterations = max_iterations;
  for ( auto &v: vertices ) {
    v.value = initialize(v.id);
    v.message = zero;
  }
  for ( auto &e: edges )
    e.message = zero;
  double time = MPI_Wtime();
  double total_time = time;
  int step = 1;
  for ( ; step <= max_iterations; step++ ) {
    // start a superstep
    outgoing.clear();
    out_msgs.clear();
    for ( auto &e: edges )
      e.message = zero;
    // calculate the edge messages from the vertex data from the previous superstep
    for ( int i = 0; i < vertices.size(); i++ ) {
      auto &v = vertices[i];
      v.active = false;
      int n = (i+1 == vertices.size()) ? edges.size() : vertices[i+1].first_edge;
      for ( int j = v.first_edge; j < n; j++ ) {
        edges[j].message = merge(edges[j].message,
                                 send(v.value,edges[j].value,n-v.first_edge));
        if (trace)
          printf("setting the message of the edge %d->%d to %0.3f\n",
                 i,edges[j].destination,edges[j].message);
      }
    }
    // put outgoing messages into a vector and sort them by the edge destination
    for ( auto e: edges )
      outgoing.push_back(tuple<int,M>(e.destination,e.message));
    sort(outgoing.begin(),outgoing.end(),
         [&](tuple<int,M> &x,tuple<int,M> &y)->bool {
             return get<0>(x) < get<0>(y);
         });
    for ( auto &v: vertices )
      v.message = zero;
    // create a thread to read incoming messages from other ranks
    thread in_thread(
       [&]()->void {
         vector<tuple<int,M>> in_msgs;
         int buffer_size = sizeof(size_t)+total_num_of_vertices*sizeof(M);
         char* buffer = new char[buffer_size];
         for ( int i = 1; i < num_of_partitions; i++ ) {
           in_msgs.clear();
           int rank = receive_data(buffer,buffer_size);
           // store the incoming messages from rank in in_msgs
           deserialize(in_msgs,buffer,buffer_size);
           // update the vertex message from incomming messages
           for ( auto &m: in_msgs ) {
             auto &v = vertices[get<0>(m)-partitions[current_partition]];
             v.message = merge(v.message,get<1>(m));
             v.active = true;
             if (trace)
               printf("received on vertex %d a message %0.3f\n",
                      v.id,v.message);
           }
         }
         delete[] buffer;
       });
    // send outgoing messages to other ranks
    int index = -1;
    int p = 0;  // partition index
    int next = (partitions.size() == 1) ? INT_MAX : partitions[1];
    M m;
    for ( auto t: outgoing ) {
      if (index < 0) {
        index = get<0>(t);
        m = get<1>(t);
        for ( int i = 0; i < partitions.size(); i++ )
          if (index >= partitions[i])
            p = i;
      } else if (get<0>(t) == index)
        m = merge(m,get<1>(t));
      else {
        if (p == current_partition) {
          auto &v = vertices[index-partitions[p]];
          if (trace)
            printf("updating the message of vertex %d to %0.3f\n",
                   index,m);
          v.active = true;
          v.message = merge(m,v.message);
        } else {
          if (trace)
            printf("outgoing message of vertex %d: %0.3f\n",
                   index,m);
          out_msgs.push_back(tuple<int,M>(index,m));
        }
        index = get<0>(t);
        m = get<1>(t);
        if (index >= next) {
          if (p != current_partition) {
            char* buffer;
            int size = serialize(out_msgs,buffer);
            send_data(p,buffer,size,0);
            out_msgs.clear();
            delete[] buffer;
          }
          p++;
          next = (p+1 == partitions.size()) ? INT_MAX : partitions[p+1];
        }
      }
    }
    // same for last partition
    if (p == current_partition) {
      auto &v = vertices[index-partitions[p]];
      if (trace)
        printf("updating the message of vertex %d to %0.3f\n",
               index,m);
      v.active = true;
      v.message = merge(m,v.message);
    } else {
      if (trace)
        printf("outgoing message of vertex %d: %0.3f\n",
               index,m);
      out_msgs.push_back(tuple<int,M>(index,m));
      char* buffer;
      int size = serialize(out_msgs,buffer);
      send_data(p,buffer,size,0);
      out_msgs.clear();
      delete[] buffer;
    }
    // update vertex values from incomming messages
    for ( int i = 0; i < vertices.size(); i++ ) {
      auto &v = vertices[i];
      if (v.active) {
        V old_value = v.value;
        v.value = new_value(v.value,v.message);
        v.active = activate(v.value,old_value);
        if (trace)
          printf("changing value of vertex %d from %0.3f to %0.3f\n",
                 v.id,old_value,v.value);
      }
    }
    in_thread.join();
    // end of a superstep
    bool active = false;
    for ( auto v: vertices )
      active = active || v.active;
    bool exit = !or_all(active);
    if (executor_rank == 0)
      printf("Step %d took %.3f secs\n",step,MPI_Wtime()-time);
    if (exit)
      break;
    time = MPI_Wtime();
  }
  if (executor_rank == 0)
    printf("End of evaluation: steps=%d, total time=%.3f secs\n",
           step-1,MPI_Wtime()-total_time);
}

// iterate over vertices and apply f to each vertex at the coordinator
template< typename ID, typename V, typename E, typename M >
void GraphPartition<ID,V,E,M>::collect ( function<void(ID,V)> f ) {
  const int coordinator = 0;
  int buffer_size = sizeof(size_t)+vertices.size()*sizeof(vertex_t)+100;
  if (executor_rank == coordinator) {
    vector<vertex_t> in_vertices;
    for ( auto &v: vertices )
      f(v.id,v.value);
    char* buffer = new char[buffer_size];
    for ( int i = 1; i < num_of_executors; i++ ) {
      in_vertices.clear();
      receive_data(buffer,buffer_size);
      deserialize(in_vertices,buffer,buffer_size);
      for ( auto &v: in_vertices )
        f(v.id,v.value);
    }
    delete[] buffer;
  } else {
    char* buffer;
    int size = serialize(vertices,buffer);
    send_data(0,buffer,size,0);
    delete[] buffer;
  }
  barrier();
}
