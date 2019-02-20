# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N abc_abias_t0.5_sigz1
#PBS -m bea
#PBS -M changhoonhahn@lbl.gov
cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda2/bin:$PATH"

export CENTRALMS_DIR="/mount/sirocco1/hahn/centralms/"
export CENTRALMS_CODEDIR="/home/users/hahn/projects/centralMS/"

tduty="0.5"
sfs="broken"

mpirun -np $NPROCS python /home/users/hahn/projects/centralMS/run/abc.py abias_z1sigma 0.5 $tduty $sfs 0.35 15 1000 > "/home/users/hahn/projects/centralMS/run/rSFH_abias0.5_"$tduty"gyr.sigma_z1_0.35.sfs"$sfs".log"
