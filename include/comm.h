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

extern int num_of_executors;
extern int executor_rank;

// send data to an MPI executor
void send_data ( int rank, char* buffer, size_t len, int tag );

// broadcast data from rank to all executors
void bcast_data ( int rank, char* buffer, size_t len );

// receive data from rank
int receive_data ( char* buffer, const size_t buffer_size, int rank );
// receive data from any MPI executor - return the sender rank
int receive_data ( char* buffer, const size_t buffer_size );

// or-together all values (blocking)
bool or_all ( bool b );

// get max of all values (blocking)
size_t max_all ( size_t n );

// barrier synchronization
void barrier ();

void start_comm ( int argc, char* argv[] );

void end_comm ();

void abort_comm ();
