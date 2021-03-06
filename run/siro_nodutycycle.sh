# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N abc_nodutycycle
#PBS -m bea
#PBS -M changhoonhahn@lbl.gov
cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda2/bin:$PATH"

export CENTRALMS_DIR="/mount/sirocco1/hahn/centralms/"
export CENTRALMS_CODEDIR="/home/users/hahn/projects/centralMS/"

sfs="flex"

mpirun -np $NPROCS python /home/users/hahn/projects/centralMS/run/abc.py nodutycycle $sfs 15 1000 > "/home/users/hahn/projects/centralMS/run/rSFH_0.2sfs_"$tduty"gyr.sfs"$sfs".log"
