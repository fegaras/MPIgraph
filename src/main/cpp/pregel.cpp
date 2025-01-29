/*
 * Copyright © 2025 University of Texas at Arlington
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

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

//#include <bitset>
#include <vector>
#include <tuple>

#include "mpi.h"
#include "omp.h"


int num_of_executors = 1;
int executor_rank = 0;

const auto comm = MPI_COMM_WORLD;

using namespace std;

template< class ID, class NV, class EV, class S >
class GraphPartition {
private:

  typedef struct {
    int first_node;             // location of the first node in nodes for this partition
    vector<bool> needed_nodes;  // if a local node is needed by this partition, its bit is on 
  } partition_element_t;
  // information about all graph partitions
  vector<partition_element_t> partitions;

  typedef struct {
    ID id;           // node id
    NV value;        // node value
    int first_edge;  // location of the first edge of this node in edges (local nodes only)
    bool active;     // is this node active?
  } node_t;
  // all graph nodes (only local nodes have outgoing edges stored)
  vector<node_t> nodes;

  typedef struct {
    int destination;  // the location of the edge destination in nodes
    EV value;         // edge value
  } edge_t;
  // contains the outgoing edges of local nodes only
  vector<edge_t> edges;

public:

  virtual S gather ( NV src_val, EV edge_val, NV dest_val ) = 0;

  virtual S sum ( S x, S y ) = 0;

  virtual NV apply ( NV node_val, S acc ) = 0;

  virtual NV scatter ( NV new_val, EV edge_val, NV dest_val ) = 0;

  void pregel () {
  }
};


class PageRank: public GraphPartition<long,float,float,float> {
};

void example () {
}


int main ( int argc, char* argv[] ) {
  // multiple threads may call MPI, with no restrictions
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    printf("Required MPI_THREAD_MULTIPLE to run MPI+OpenMP");
    MPI_Finalize();
    exit(-1);
  }
  MPI_Comm_set_errhandler(comm,MPI_ERRORS_RETURN);
  MPI_Comm_rank(comm,&executor_rank);
  MPI_Comm_size(comm,&num_of_executors);
  char machine_name[256];
  gethostname(machine_name,255);
  int local_rank = 0;
  char* lc = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (lc == nullptr)
    lc = getenv("MV2_COMM_WORLD_LOCAL_RANK");
  if (lc != nullptr)
    local_rank = atoi(lc);
  int ts;
  #pragma omp parallel
  { ts = omp_get_num_threads(); }
  printf("Using executor %d: %s/%d (threads: %d)\n",
         executor_rank,machine_name,local_rank,ts);


  MPI_Finalize();
}
