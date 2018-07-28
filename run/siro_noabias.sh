# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N abc_noabias
#PBS -m bea
#PBS -M changhoonhahn@lbl.gov
cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda2/bin:$PATH"

export CENTRALMS_DIR="/mount/sirocco1/hahn/centralms/"
export CENTRALMS_CODEDIR="/home/users/hahn/projects/centralMS/"

tduty="0.5"
sfs="flex"

mpirun -np $NPROCS python /home/users/hahn/projects/centralMS/run/abc.py noabias $tduty $sfs 15 1000 > "/home/users/hahn/projects/centralMS/run/randomSFH"$tduty"gyr.sfs"$sfs".log"
