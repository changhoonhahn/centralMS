# !/bin/bash
# PBS -l nodes=1:ppn=24
# PBS -N randomSFH_5gyr
cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda2/bin:$PATH"

export CENTRALMS_DIR="/mount/sirocco1/hahn/centralms/"
export CENTRALMS_CODEDIR="/home/users/hahn/projects/centralMS/"

tduty=5

/usr/local/openmpi-1.10.7/bin/mpiexec -np $NPROCS -npernode 24 python /home/users/hahn/projects/centralMS/run/abc.py noabias $tduty 15 1000 > /home/users/hahn/projects/centralMS/run/randomSFH_5gyr.log
