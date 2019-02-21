# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N abc_r0.0_t0.5_sigz1
#PBS -m bea
#PBS -M changhoonhahn@lbl.gov
cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda2/bin:$PATH"

export CENTRALMS_DIR="/mount/sirocco1/hahn/centralms/"
export CENTRALMS_CODEDIR="/home/users/hahn/projects/centralMS/"

# this script is to generate the abc posteriors for testing what
# happens when we change sigma_logM* at z=1. 

tduty="0.5"
sfs="broken"

# no assembly bias (i.e. r = 0) 
mpirun -np $NPROCS python /home/users/hahn/projects/centralMS/run/abc.py noabias_z1sigma $tduty $sfs 0.35 15 1000 > "/home/users/hahn/projects/centralMS/run/rSFH_noabias_"$tduty"gyr.sigma_z1_0.35.sfs"$sfs".log"
 
# yes assembly bias (i.e. r > 0) 
#mpirun -np $NPROCS python /home/users/hahn/projects/centralMS/run/abc.py abias_z1sigma 0.99 $tduty $sfs 0.35 15 1000 > "/home/users/hahn/projects/centralMS/run/rSFH_abias0.99_"$tduty"gyr.sigma_z1_0.35.sfs"$sfs".log"
