#!/bin/sh
#
# stdout/stderr redirection
#PBS -N single_dbn_PCD_small_3xlearning_mom
#PBS -o $PBS_JOBNAME.$PBS_JOBID.out
#PBS -e $PBS_JOBNAME.$PBS_JOBID.err
#PBS -M skaaesonderby@gmail.com
#PBS -m abe
# Execute the job from the current working directory
cd $PBS_O_WORKDIR

matlab -nodisplay -r "test_dbn7mom" 
