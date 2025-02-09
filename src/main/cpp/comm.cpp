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

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "mpi.h"

using namespace std;

int num_of_executors = 1;
int executor_rank = 0;

const bool trace = false;
const auto comm = MPI_COMM_WORLD;

// send data to an MPI executor
void send_data ( int rank, char* &buffer, size_t len, int tag ) {
  MPI_Send(buffer,len,MPI_BYTE,rank,tag,comm);
  if (trace)
    printf("Executor %d: sent %ld bytes to %d\n",
           executor_rank,len,rank);
}

// broadcast data to all executors
void bcast_data ( char* &buffer, const size_t buffer_size ) {
  MPI_Bcast(buffer,buffer_size,MPI_BYTE,executor_rank,comm);
  if (trace)
    printf("Executor %d: broadcast %ld bytes\n",
           executor_rank,buffer_size);
}

// receive data from any MPI executor - return the executor rank
int receive_data ( char* &buffer, const size_t buffer_size ) {
  MPI_Status status;
  MPI_Recv(buffer,buffer_size,MPI_BYTE,MPI_ANY_SOURCE,MPI_ANY_TAG,comm,&status);
  if (trace) {
    int count;
    MPI_Get_count(&status,MPI_BYTE,&count);
    printf("Executor %d: received %d bytes from %d\n",
           executor_rank,count,status.MPI_SOURCE);
  }
  return status.MPI_SOURCE;
}

// or-together all values (blocking)
bool or_all ( bool b ) {
  int in[1] = { ( b ? 1 : 0 ) };
  int ret[1];
  MPI_Allreduce(in,ret,1,MPI_INT,MPI_LOR,comm);
  return ret[0] == 1;
}

// barrier synchronization
void barrier () {
  MPI_Barrier(comm);
  if (trace && executor_rank == 0)
    printf("Barrier synchronization\n");
}

// start MPI communication
void start_comm ( int argc, char* argv[] ) {
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    cerr << "Required MPI_THREAD_MULTIPLE\n";
    MPI_Finalize();
    exit(-1);
  }
  MPI_Comm_set_errhandler(comm,MPI_ERRORS_RETURN);
  MPI_Comm_rank(comm,&executor_rank);
  MPI_Comm_size(comm,&num_of_executors);
}

// end MPI communication
void end_comm () {
  MPI_Finalize();
}

// abort MPI communication
void abort_comm () {
  MPI_Abort(comm,-1);
}
