#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=trevor.keller@nist.gov
#SBATCH --partition mml        # partition (mml or preemptible)
#SBATCH --constraint="haswell" # Haswell
#SBATCH -t 24:00:00            # time (hh:mm:ss or dd-hh:mm:ss)
#SBATCH --nodes=8              # total machines
#SBATCH --ntasks=16            # total MPI ranks, 2 per machine
#SBATCH --cpus-per-task=16     # OMP threads per MPI rank
#SBATCH --overcommit           # unlock every available thread
#SBATCH -J TKR4p074
#SBATCH -o /wrk/tnk10/phase-field/alloy625/TKR4p074.log
#SBATCH -D /wrk/tnk10/phase-field/alloy625/TKR4p074
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# SLURM batch script for Inconel 625 coarsening simulation
#
# Make sure your binary has been compiled before launching!
# Usage:
#        module load mpi/openmpi-x86_64
#        make -B smpi
#        sbatch raritan_run.sh

# Hardware notes: Raritan Haswells are Xeon E5-2630v3,
# 8 cores per chip, 2 sockets per machine = 32 threads total.
# Use a minimum of 1 MPI rank per socket.

SRCDIR=/home/tnk10/research/projects/phase-field/Zhou718
WRKDIR=/wrk/tnk10/phase-field/alloy625/TKR4p074
SCRIPT=smpi

ALLTIME=10000000
CHKTIME=100000

# OpenMPI reads SLURM variables directly, so no flags are needed.
mpirun --contiguous --bind-to core $SRCDIR/./$SCRIPT --example 2 $WRKDIR/superalloy.dat
mpirun --contiguous --bind-to core $SRCDIR/./$SCRIPT $WRKDIR/superalloy.dat $ALLTIME $CHKTIME

# PS: Don't use srun with OpenMPI.
