#!/bin/bash
#SBATCH -p ccb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=7-00:00:00
#SBATCH --mem=512GB
#SBATCH --job-name=bingham
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sweady@flatironinstitute.org
#SBATCH --output=bingham_%j.out

# Clean up
module purge 

# Location of initial data
INITIAL_DATA="none"

# Set parameter file
PARAMETER_FILE=params.h

# Create run directory in ceph 
RUNDIR=~/ceph/bingham-${SLURM_JOB_ID}
mkdir $RUNDIR

# Compile
make clean
make params=$PARAMETER_FILE

# Move compiled code into run folder
mv main $RUNDIR/

# Copy parameters for safekeeping
cp $PARAMETER_FILE $RUNDIR/params_${SLURM_JOB_ID}.h 

# Copy initial data if it exists, otherwise new initial condition 
if [ -d "$INITIAL_DATA" ]; then
  cp -r "$INITIAL_DATA" $RUNDIR/initial_data
  RESUME=1 
else
  RESUME=0
fi

# Move to run directory
cd $RUNDIR

# Start
./main 120 $RESUME

