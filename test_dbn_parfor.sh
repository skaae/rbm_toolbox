#!/bin/sh
#
# stdout/stderr redirection
#PBS -N rbm_eval
#PBS -o $PBS_JOBNAME.$PBS_JOBID.out
#PBS -e $PBS_JOBNAME.$PBS_JOBID.err
#PBS -M skaaesonderby@gmail.com
#PBS -m abe
#PBS -l nodes=1:ppn=12
# Execute the job from the current working directory
cd $PBS_O_WORKDIR

matlab -nodisplay -r "test_dbn_parfor" 
