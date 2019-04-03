# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N modelabc 

cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda2/bin:$PATH"

source /home/users/hahn/.bashrc

niter=14
for rcorr in 0.5 0.99; do #0.; do #
    for tduty in "0.5" "1" "2" "5" "10"; do 
        if (( $(echo "$rcorr > 0." |bc -l) )); then 
            run="rSFH_abias"$rcorr"_"$tduty"gyr.sfsmf.sigma_z1_0.45.sfsbroken"
        else
            run="randomSFH"$tduty"gyr.sfsmf.sigma_z1_0.45.sfsbroken"
        fi 
        echo $run
        mpirun -np $NPROCS python /home/users/hahn/projects/centralMS/run/modelabc_sigmaz1.py $run 0.45 $niter > "/home/users/hahn/projects/centralMS/run/"$run"."$niter".modelabc.log"
    done 
done 
