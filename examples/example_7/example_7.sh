#!/bin/sh
#
# stdout/stderr redirection
#PBS -N dbn_example_7_sparse_hybrid
#PBS -o $PBS_JOBNAME.$PBS_JOBID.out
#PBS -e $PBS_JOBNAME.$PBS_JOBID.err
#PBS -l nodes=1:ppn=1,gpus=1;
# Execute the job from the current working directory
cd $PBS_O_WORKDIR
ML_GPUDEVICE=`sed 's/^.*gpu//' $PBS_GPUFILE`
export ML_GPUDEVICE 

echo -n gpu${ML_GPUDEVICE}@/bin/hostname
 
matlab -nodisplay -r "dbn_example_7" 