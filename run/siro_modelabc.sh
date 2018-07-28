# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N modelabc 
#PBS -m bea
#PBS -M changhoonhahn@lbl.gov
#PBS -e /home/users/hahn/projects/centralMS/run/modelabc.e
#PBS -o /home/users/hahn/projects/centralMS/run/modelabc.o

cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda2/bin:$PATH"

source /home/users/hahn/.bashrc
#export CENTRALMS_DIR="/mount/sirocco1/hahn/centralms/"
#export CENTRALMS_CODEDIR="/home/users/hahn/projects/centralMS/"

run="randomSFH1gyr.sfsanchored"
niter=12
echo $NPROCS > /home/users/hahn/projects/centralMS/run/blah.txt
mpirun -np 24 python /home/users/hahn/projects/centralMS/run/abc.py modelrun $run $niter > "/home/users/hahn/projects/centralMS/run/"$run"."$niter".modelabc.log"
