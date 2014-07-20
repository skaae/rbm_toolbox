#!/bin/bash
folder="mnist_cRBM_CD"
mkdir -p "$folder"
for count in {1..72}
do
    echo "#!/bin/sh
#
# stdout/stderr redirection
#PBS -N $folder.$count
#PBS -o $folder/\$PBS_JOBNAME.$PBS_JOBID.out
#PBS -e $folder/\$PBS_JOBNAME.$PBS_JOBID.err
#PBS -l nodes=1:ppn=1
#Execute the job from the current working directory
cd \$PBS_O_WORKDIR
matlab -nodesktop -r \"t=$count;folder='$folder'; test_dbn_parfor\" " > experiment.sh

qsub experiment.sh
done
