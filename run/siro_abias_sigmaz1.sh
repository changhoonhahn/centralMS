# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N abc_abias_sigz1
#PBS -e /home/users/hahn/projects/centralMS/run/rSFH_abias_sigz1.log
#PBS -M changhoonhahn@lbl.gov
cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda2/bin:$PATH"

export CENTRALMS_DIR="/mount/sirocco1/hahn/centralms/"
export CENTRALMS_CODEDIR="/home/users/hahn/projects/centralMS/"

tduty="10"
sfs="broken"

mpirun -np $NPROCS python /home/users/hahn/projects/centralMS/run/abc.py abias_z1sigma 0.5 $tduty $sfs 0.35 1 100 
