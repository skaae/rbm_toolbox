#!/bin/sh
#
# stdout/stderr redirection
#PBS -N dbn_example_9
#PBS -o $PBS_JOBNAME.$PBS_JOBID.out
#PBS -e $PBS_JOBNAME.$PBS_JOBID.err
#PBS -l nodes=1:ppn=1
# Execute the job from the current working directory
cd $PBS_O_WORKDIR

matlab -nodisplay -r "dbn_example_9"