#!/bin/sh
#
# stdout/stderr redirection
#PBS -N multilayer_dbn_CD
#PBS -o $PBS_JOBNAME.$PBS_JOBID.out
#PBS -e $PBS_JOBNAME.$PBS_JOBID.err
#PBS -M skaaesonderby@gmail.com
#PBS -m abe
# Execute the job from the current working directory
cd $PBS_O_WORKDIR

matlab -nodisplay -r "test_dbn4" 
