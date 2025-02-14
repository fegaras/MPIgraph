#!/bin/bash
#SBATCH -A uot166
#SBATCH --job-name="MPIgraph"
#SBATCH --output="run.log"
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --mem=249208M
#SBATCH --export=ALL
#SBATCH --time=30    # time limit in minutes

module purge
module load slurm cpu/0.17.3b gcc/10.2.0/npcyll4 openmpi/4.1.3

mpic++ -O2 -std=c++11 -DNDEBUG -Iinclude src/main/cpp/*.cpp -o pregel

ulimit -l unlimited
ulimit -s unlimited

cores=32

mpirun -N $cores --bind-to socket pregel 100000000 10
