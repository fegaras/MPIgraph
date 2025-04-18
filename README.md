# MPIgraph

Implementation of the Pregel algorithm for large graph processing using MPI.

### Installation

MPIgraph depends on MPI, C++ (11 or newer), and make.
Install either
open-mpi 5.0 from [https://www.open-mpi.org/software/](https://www.open-mpi.org/software/) or
MVAPICH2 2.3 from [https://mvapich.cse.ohio-state.edu/downloads/](https://mvapich.cse.ohio-state.edu/downloads/).

Edit the file `setup.sh` to point to your installation directories.
For open-mpi, do:
```bash
source setup.sh
```
for MVAPICH2, do:
```bash
mvapich=y source setup.sh
```
Compile MPIgraph using:
```bash
make
```

You can test Pregel by running PageRank on a random graph of 1000 vertices using two executors and 5 steps using:
```bash
n=2 ./run 1000 5
```
or on the [Google web graph](https://snap.stanford.edu/data/web-Google.html) using:
```bash
n=2 ./run web-Google.txt 5
```
