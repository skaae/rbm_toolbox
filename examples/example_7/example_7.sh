#!/bin/sh
#
# stdout/stderr redirection
#PBS -N dbn_example_7
#PBS -o $PBS_JOBNAME.$PBS_JOBID.out
#PBS -e $PBS_JOBNAME.$PBS_JOBID.err
#PBS -l nodes=1:ppn=2
#PBS -l walltime=72:00:00
# Execute the job from the current working directory
cd $PBS_O_WORKDIR

matlab -nodisplay -r "dbn_example_7"