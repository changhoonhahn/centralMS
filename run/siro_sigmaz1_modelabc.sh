# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N modelabc 

cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda2/bin:$PATH"

source /home/users/hahn/.bashrc

niter=14
for rcorr in 0.5; do #0.99 0.
    for tduty in "10"; do #"0.5" "1" "2" "5"; do # "10"; do 
        run="rSFH_abias"$rcorr)"_"$tduty"gyr.sfsmf.sigma_z1_0.35.sfsbroken"
        echo $run
        mpirun -np $NPROCS python /home/users/hahn/projects/centralMS/run/modelabc_sigmaz1.py $run 0.35 $niter > "/home/users/hahn/projects/centralMS/run/"$run"."$niter".modelabc.log"
    done 
done 
