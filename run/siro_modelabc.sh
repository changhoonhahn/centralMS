# !/bin/bash
# PBS -l nodes=1:ppn=24
cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda2/bin:$PATH"

export CENTRALMS_DIR="/mount/sirocco1/hahn/centralms/"
export CENTRALMS_CODEDIR="/home/users/hahn/projects/centralMS/"

run="randomSFH2gyr.sfsanchored"
niter=12

mpirun -np 24 python /home/users/hahn/projects/centralMS/run/abc.py modelrun $run $niter > "/home/users/hahn/projects/centralMS/run/"$run"."$niter".modelabc.log"
