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
#include <mutex>
#include <stdarg.h>
#include "mpi.h"

#include "serialization.h"
#include "comm.h"

using namespace std;

// tracing works for long ID and float V and M only
#define trace false
#ifdef trace
#define info(...) { info_(__VA_ARGS__); }
#else
#define info(...) { }
#endif

/* A graph partition of an executor. A partition is associated with a set of vertices 
   that do not overlap with the vertices in other partitions. It also contains all
   edges whose source is one of these vertices and all sinks from these vertices.
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
  vector<long> partitions;

  // # of vertices in each partition
  vector<size_t> partition_vertices;

  size_t total_num_of_vertices;
  size_t max_num_of_vertices;

  // message send to all vertices
  M global;

  typedef struct {
    ID id;            // vertex id
    V value;          // vertex value
    M message;        // the incoming message to this vertex
    long first_edge;  // index of the first edge of this vertex (in edges)
    bool active;      // is this vertex active?
  } vertex_t;
  // local vertices
  vector<vertex_t> vertices;

  typedef struct {
    long destination; // the index of the edge destination
    E value;          // edge value
    M message;        // the outgoing message to be send to the edge destination
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
  long id_loc ( ID id ) const;

  // get the vertex id at index loc
  ID get_id ( long loc ) const;

  void send_global_only ( M global, int from_rank, int to_rank ) const;

  // print tracing info
  void info_ ( const char* fmt, ... ) const {
    static mutex info_mutex;
    if (trace) {
      lock_guard<mutex> lock(info_mutex);
      va_list args;
      va_start(args,fmt);
      printf("[%2d] ",executor_rank);
      vprintf(fmt,args);
      printf("\n");
      va_end(args);
    }
  }

public:

  GraphPartition () {
    num_of_partitions = num_of_executors;
    current_partition = executor_rank;
    partitions = vector<long>(num_of_partitions+1,0);
    tmp_vertices = new vector<vector<ID>>();
    tmp_sinks = new vector<ID>();
    for ( int i = 0; i < num_of_partitions; i++ )
      tmp_vertices->push_back(vector<ID>());
    tmp_edges = new vector<tuple<ID,ID,E>>();
  }

  // add a graph edge
  void add_edge ( ID from, ID to, E edge_val ) {
    info("adding edge %ld -> %ld",from,to);
    tmp_edges->push_back(tuple<ID,ID,E>(from,to,edge_val));
  }

  // add a sink (a vertex that doesn't have any out-neighbors)
  void add_sink ( ID vertex ) {
    info("adding sink %ld",vertex);
    tmp_sinks->push_back(vertex);
  }

  inline size_t num_of_vertices () const {
    return total_num_of_vertices;
  }

  // redistribute edges and sinks based on the hash function
  void partition ( function<int(ID)> hash );

  // build the graph partitions from the edges and the sinks
  void build_graph ();

  // initialize a vertex
  virtual V initialize ( ID id ) const = 0;

  // the zero value of merge
  M zero;

  // merge messages
  virtual M merge ( M x, M y ) const = 0;

  // calculate a new vertex value from the current vertex value
  //  and from the merged messages
  virtual V new_value ( V val, M acc ) const = 0;

  // calculate the message value to send to the edge destination;
  //   new_val is the new vertex value of the edge source;
  //   degree is the number of out-neighbors
  virtual M send ( V new_val, E edge_val, int degree ) const = 0;

  // calculate the message value to send to all vertices (return zero to ignore)
  virtual M send_all ( V new_val, int degree ) const = 0;

  // if true, activate this vertex at the next superstep
  virtual bool activate ( V old_val, V new_val ) const = 0;

  // Pregel graph processing
  void pregel ( int max_iterations );

  // iterate over vertices and apply f to each vertex at the coordinator
  void foreach ( function<void(ID,V)> f );
};

// get the index of a vertex id
template< typename ID, typename V, typename E, typename M >
long GraphPartition<ID,V,E,M>::id_loc ( ID id ) const {
  for ( int i = 0; i < num_of_partitions; i++ ) {
    auto first = (*tmp_vertices)[i].begin();
    auto last = (*tmp_vertices)[i].end();
    auto it = lower_bound(first,last,id);
    if (it != last && id >= *it)
      return (it-first)+partitions[i];
  }
  return 0;
}

// get the vertex id at index loc
template< typename ID, typename V, typename E, typename M >
ID GraphPartition<ID,V,E,M>::get_id ( long loc ) const {
  for ( int i = 0; i < num_of_partitions; i++ )
    if (loc < partitions[i])
      return (*tmp_vertices)[i-1][loc-partitions[i-1]];
  return (*tmp_vertices)[num_of_partitions-1]
                [loc-partitions[num_of_partitions-1]];
}

template< typename ID, typename V, typename E, typename M >
void GraphPartition<ID,V,E,M>::send_global_only
                 ( M global, int from_rank, int to_rank ) const {
  if (from_rank < to_rank) {
    vector<tuple<long,M>> out_msgs;
    if (global != zero)
      out_msgs.push_back(tuple<long,M>(-1,global));
    char* buffer;
    size_t size = serialize(out_msgs,buffer);
    for ( int i = from_rank; i < to_rank; i++ )
      if (i != current_partition) {
        info("sending an empty vector to node %d",i);
        send_data(i,buffer,size,0);
      }
    out_msgs.clear();
    delete[] buffer;
  }
}

// distribute edges based on the hash function and find sinks
template< typename ID, typename V, typename E, typename M >
void GraphPartition<ID,V,E,M>::partition ( function<int(ID)> hash ) {
  double time = MPI_Wtime();
  vector<vector<ID>> out_destinations;
  vector<vector<tuple<ID,ID,E>>> out_edges;
  long num_of_edges = tmp_edges->size();
  for ( int i = 0; i < num_of_partitions; i++ ) {
    out_destinations.push_back(vector<ID>());
    out_edges.push_back(vector<tuple<ID,ID,E>>());
  }
  while (!tmp_edges->empty()) {
    auto t = tmp_edges->back();
    tmp_edges->pop_back();
    out_edges[abs(hash(get<0>(t)))%num_of_partitions].push_back(t);
    out_destinations[abs(hash(get<1>(t)))%num_of_partitions].push_back(get<1>(t));
  }
  delete tmp_edges;
  tmp_edges = new vector<tuple<ID,ID,E>>(out_edges[current_partition]);
  size_t max_size = 0;
  for ( int i = 0; i < num_of_partitions; i++ )
    if (i != current_partition)
      max_size = max(max_size,max(out_edges[i].size(),out_destinations[i].size()));
  max_size = max_all(max_size);
  thread trcv(
      [&]()->void {
        size_t buffer_size = sizeof(size_t)+max_size*sizeof(tuple<ID,ID,E>);
        char* buffer = new char[buffer_size];
        for ( int i = 1; i < num_of_partitions; i++ ) {
          receive_data(buffer,buffer_size);
          deserialize(out_destinations[current_partition],buffer,buffer_size);
        }
        for ( int i = 1; i < num_of_partitions; i++ ) {
          receive_data(buffer,buffer_size);
          deserialize(*tmp_edges,buffer,buffer_size);
        }
        delete[] buffer;
      });
  char* buffer;
  for ( int i = 0; i < num_of_partitions; i++ )
    if (i != current_partition) {
      size_t size = serialize(out_destinations[i],buffer);
      send_data(i,buffer,size,0);
      delete[] buffer;
    }
  barrier();
  for ( int i = 0; i < num_of_partitions; i++ )
    if (i != current_partition) {
      size_t size = serialize(out_edges[i],buffer);
      send_data(i,buffer,size,0);
      delete[] buffer;
    }
  trcv.join();
  tmp_sinks = new vector<ID>();
  auto ds = out_destinations[current_partition];
  sort(ds.begin(),ds.end());
  ds.erase(unique(ds.begin(),ds.end()),ds.end());
  sort(tmp_edges->begin(),tmp_edges->end(),
       [&](tuple<ID,ID,E> x,tuple<ID,ID,E> y)->bool {
           return get<0>(x) < get<0>(y);
       });
  auto t = tuple<ID,ID,E>();
  for ( auto &d: ds ) {
    get<0>(t) = d;
    if (!binary_search(tmp_edges->begin(),tmp_edges->end(),t,
		       [&](tuple<ID,ID,E> x,tuple<ID,ID,E> y)->bool {
			 return get<0>(x) < get<0>(y);
		       }))
      tmp_sinks->push_back(d);
  }
  printf("Executor %d: sinks=%ld, edges=%ld\n",
         executor_rank,tmp_sinks->size(),tmp_edges->size());
  barrier();
  if (executor_rank == 0)
    printf("Graph partition time = %.3f secs\n",MPI_Wtime()-time);
}

// build the graph partition from the sorted edges and sinks
template< typename ID, typename V, typename E, typename M >
void GraphPartition<ID,V,E,M>::build_graph () {
  double time = MPI_Wtime();
  ID vid;
  long index = 0;
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
  // broadcast the # of vertices in each partition to all ranks
  partition_vertices.resize(num_of_partitions);
  partition_vertices[current_partition] = (*tmp_vertices)[current_partition].size();
  barrier();
  for ( int i = 0; i < num_of_partitions; i++ )
    bcast_data(i,(char*)&(partition_vertices[i]),sizeof(size_t));
  // broadcast the local vertices to all ranks
  char* cbuffer;
  serialize((*tmp_vertices)[current_partition],cbuffer);
  for ( int i = 0; i < num_of_partitions; i++ ) {
    size_t buffer_size = sizeof(size_t)+partition_vertices[i]*sizeof(ID);
    char* buffer = (i == current_partition) ? cbuffer : new char[buffer_size];
    bcast_data(i,buffer,buffer_size);
    if (i != current_partition)
      deserialize((*tmp_vertices)[i],buffer,buffer_size);
    delete[] buffer;
  }
  // build the partition table
  size_t n = 0;
  int i = 0;
  for ( auto size: partition_vertices ) {
    n += size;
    partitions[++i] = n;
  }
  total_num_of_vertices = n;
  max_num_of_vertices = 0;
  for ( int i = 0; i < num_of_partitions; i++ )
    if (i != current_partition)
      max_num_of_vertices = max(max_num_of_vertices,partition_vertices[i]);
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
  if (!trace)
    delete tmp_vertices;
  delete tmp_edges;
  barrier();
  if (executor_rank == 0)
    printf("Graph build time = %.3f secs\n",MPI_Wtime()-time);
}

// Pregel graph processing
template< typename ID, typename V, typename E, typename M >
void GraphPartition<ID,V,E,M>::pregel ( int max_iterations ) {
  vector<tuple<long,M>> outgoing;
  vector<tuple<long,M>> out_msgs;
  for ( auto &v: vertices ) {
    v.value = initialize(v.id);
    v.message = zero;
  }
  global = zero;
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
    for ( long i = 0; i < vertices.size(); i++ ) {
      auto &v = vertices[i];
      v.active = false;
      long n = (i+1 == vertices.size()) ? edges.size() : vertices[i+1].first_edge;
      global = merge(global,send_all(v.value,n-v.first_edge));
      for ( long j = v.first_edge; j < n; j++ ) {
        edges[j].message = merge(edges[j].message,
                                 send(v.value,edges[j].value,n-v.first_edge));
        info("setting message of edge %ld->%ld to %0.3f",
             v.id,get_id(edges[j].destination),edges[j].message);
      }
    }
    // put outgoing messages into a vector and sort them by the edge destination
    for ( auto &e: edges )
      outgoing.push_back(tuple<long,M>(e.destination,e.message));
    sort(outgoing.begin(),outgoing.end(),
         [&](tuple<long,M> &x,tuple<long,M> &y)->bool {
             return get<0>(x) < get<0>(y);
         });
    for ( auto &v: vertices )
      v.message = zero;
    // create a thread to read incoming messages from other ranks
    thread in_thread(
       [&]()->void {
         vector<tuple<long,M>> in_msgs;
         size_t buffer_size = sizeof(size_t)+max_num_of_vertices*sizeof(tuple<long,M>);
         char* buffer = new char[buffer_size];
         for ( int i = 1; i < num_of_partitions; i++ ) {
           in_msgs.clear();
           int rank = receive_data(buffer,buffer_size);
           // store the incoming messages from rank in in_msgs
           deserialize(in_msgs,buffer,buffer_size);
           // update the vertex message from incomming messages
           for ( auto &m: in_msgs )
	     if (get<0>(m) < 0) {
	       // received a send_all message
	       for ( int i = 0; i < vertices.size(); i++ )
		 vertices[i].message = merge(vertices[i].message,get<1>(m));
	     } else {
	       auto &v = vertices[get<0>(m)-partitions[current_partition]];
	       v.message = merge(v.message,get<1>(m));
	       v.active = true;
	       info("received and merged on vertex %ld a message %0.3f",
		    v.id,v.message);
	     }
         }
         delete[] buffer;
       });
    // send outgoing messages to other ranks
    if (outgoing.size() == 0) {
      // nothing to send
      out_msgs.clear();
      char* buffer;
      size_t size = serialize(out_msgs,buffer);
      for ( int i = 0; i < num_of_partitions; i++ )
        if (i != current_partition)
          send_data(i,buffer,size,0);
      out_msgs.clear();
      delete[] buffer;
    } else {
      long index = -1;  // vertex loc
      int p = 0;  // partition index
      long next = partitions[1];
      M m;
      for ( auto &t: outgoing ) {
        if (index < 0) {
          // at the first edge only
          index = get<0>(t);
          m = get<1>(t);
          for ( int i = 0; i < num_of_partitions; i++ )
            if (index >= partitions[i])
              p = i;
          send_global_only(global,0,p);
        } else if (get<0>(t) == index) {
          // merge consecutive messages to the same vertex
          m = merge(m,get<1>(t));
        } else {
          // start processing a new vertex
          if (p == current_partition) {
            auto &v = vertices[index-partitions[p]];
            v.active = true;
            v.message = merge(m,v.message);
            info("updating the message of vertex %ld to %0.3f",
                 v.id,v.message);
          } else {
            info("outgoing message to vertex %ld at rank %d: %0.3f",
                 get_id(index),p,m);
            out_msgs.push_back(tuple<long,M>(index,m));
          }
          index = get<0>(t);
          m = get<1>(t);
          if (index >= next) {
            // end of partition p
            if (p == current_partition) {
              // merge global message
              if (global != zero)
                for ( int i = 0; i < vertices.size(); i++ ) {
                  vertices[i].active = true;
                  vertices[i].message = merge(vertices[i].message,global);
                }
            } else {
              if (global != zero)
                // send also the global message
                out_msgs.push_back(tuple<long,M>(-1,global));
              char* buffer;
              size_t size = serialize(out_msgs,buffer);
              send_data(p,buffer,size,0);
              out_msgs.clear();
              delete[] buffer;
            }
            int old_p = p;
            for ( int i = p+1; i < num_of_partitions; i++ )
              if (index >= partitions[i])
                p = i;
            send_global_only(global,old_p+1,p);
            next = partitions[p+1];
          }
        }
      }
      { // same for last partition
        auto &v = vertices[index-partitions[p]];
        if (p == current_partition) {
          info("updating the message of vertex %ld to %0.3f",v.id,m);
          v.active = true;
          v.message = merge(m,v.message);
          if (global != zero)
            for ( int i = 0; i < vertices.size(); i++ )
              vertices[i].message = merge(vertices[i].message,global);
        } else {
          info("outgoing message to vertex %ld at rank %d: %0.3f",
               get_id(index),p,m);
          out_msgs.push_back(tuple<long,M>(index,m));
          if (global != zero)
            // send also the global message
            out_msgs.push_back(tuple<long,M>(-1,global));
          char* buffer;
          size_t size = serialize(out_msgs,buffer);
          send_data(p,buffer,size,0);
          out_msgs.clear();
        }
        send_global_only(global,p+1,num_of_partitions);
      }
    }
    in_thread.join();
    // update vertex values from incomming messages
    for ( long i = 0; i < vertices.size(); i++ ) {
      auto &v = vertices[i];
      if (v.active) {
        V old_value = v.value;
        v.value = new_value(v.value,v.message);
        v.active = activate(v.value,old_value);
        info("changing value of vertex %ld from %0.3f to %0.3f",
             v.id,old_value,v.value);
      }
    }
    // end of a superstep
    if (executor_rank == 0)
      printf("Step %d took %.3f secs\n",step,MPI_Wtime()-time);
    bool active = false;
    for ( auto &v: vertices )
      active = active || v.active;
    // barrier synchronization
    if (!or_all(active)) {
      step++;
      break;
    }
    global = zero;
    time = MPI_Wtime();
  }
  if (executor_rank == 0)
    printf("End of evaluation: steps=%d, total time=%.3f secs\n",
           step-1,MPI_Wtime()-total_time);
}

// iterate over vertices and apply f to each vertex at the coordinator
template< typename ID, typename V, typename E, typename M >
void GraphPartition<ID,V,E,M>::foreach ( function<void(ID,V)> f ) {
  const int coordinator = 0;
  if (executor_rank == coordinator) {
    vector<vertex_t> in_vertices;
    for ( auto &v: vertices )
      f(v.id,v.value);
    size_t buffer_size = sizeof(size_t)+max_num_of_vertices*sizeof(vertex_t);
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
    size_t buffer_size = serialize(vertices,buffer);
    send_data(coordinator,buffer,buffer_size,0);
    delete[] buffer;
  }
  barrier();
}
