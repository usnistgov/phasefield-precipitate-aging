#!/usr/bin/bash
#SBATCH --mail-type=ARRAY_TASKS,END
#SBATCH --mail-user=trevor.keller@nist.gov
#SBATCH --partition preemptible # partition (mml or preemptible)
#SBATCH --time 96:00:00         # -t, time (hh:mm:ss or dd-hh:mm:ss)
#SBATCH --ntasks=30             # -n, total number of MPI ranks (1 per socket)
#SBATCH --cpus-per-task=16      # -c, 16 OMP threads per MPI rank
#SBATCH --array=20-23           # spawn four jobs
#SBATCH --constraint="haswell"  # Haswell nodes (slightly faster than average)
#SBATCH --contiguous
#SBATCH -J TKR4p119
#SBATCH -o /wrk/tnk10/phase-field/alloy625/TKR4p119/run%a.log
#SBATCH -D /wrk/tnk10/phase-field/alloy625/TKR4p119
export OMP_NUM_THREADS=16

# SLURM batch script for Inconel 625 coarsening simulation
#
# Make sure your binary has been compiled before launching!
# Before launching, module load PrgEnv-intel
# Usage: sbatch --export=ALL raritan_TKR4p119.sh
#
# HARDWARE DETAILS
# haswell: Intel Xeon E5-2697v3 @ 2.4GHz, 2 sockets, 16 cores, 32 threads per node, 150+ nodes

SRCDIR=/home/tnk10/research/projects/phase-field/Zhou718
WRKDIR=/wrk/tnk10/phase-field/alloy625/TKR4p119/run${SLURM_ARRAY_TASK_ID}
cd $WRKDIR

SCRIPT=ismpi

ALLTIME=50000000
CHKTIME=1000000

if [[ ! -f $SRCDIR/$SCRIPT ]]
then
	echo "Error: ${SRCDIR}/${SCRIPT} not found: cd ${SRCDIR} && make ${SCRIPT}"
else
	sleep ${SLURM_ARRAY_TASK_ID}
	mpirun --mca mpi_leave_pinned 0 --mca btl openib,self,vader $SRCDIR/./$SCRIPT --example 2 $WRKDIR/superalloy.dat
	mpirun --mca mpi_leave_pinned 0 --mca btl openib,self,vader $SRCDIR/./$SCRIPT $WRKDIR/superalloy.dat $ALLTIME $CHKTIME
fi


# If the job crashes, restart from the lastest checkpoint:
#LATEST=$(ls -t1 superalloy*.dat | head -n 1)
#mpirun --mca mpi_leave_pinned 0 --mca btl openib,self,vader $SRCDIR/./$SCRIPT $WRKDIR/$LATEST $ALLTIME $CHKTIME
