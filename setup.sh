#/bin/bash

export MPIgraph_HOME=${HOME}/MPIgraph

if [ "$mvapich" == "y" ]; then
    export mvapich
    # install MVAPICH2 2.3.7 from https://mvapich.cse.ohio-state.edu/downloads/
    export MPI_HOME=${HOME}/mvapich
else
    unset mvapich
    # install open-mpi from https://www.open-mpi.org/software/
    export MPI_HOME=${HOME}/openmpi
fi

export PATH="$MPI_HOME/bin:$MPIgraph_HOME/bin:$PATH"

export LD_LIBRARY_PATH="$MPI_HOME/lib:$LD_LIBRARY_PATH"
