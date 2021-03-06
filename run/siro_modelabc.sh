# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N modelabc 

cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda2/bin:$PATH"

source /home/users/hahn/.bashrc
#export CENTRALMS_DIR="/mount/sirocco1/hahn/centralms/"
#export CENTRALMS_CODEDIR="/home/users/hahn/projects/centralMS/"

niter=14
#tscale="0.5"
#run="randomSFH0.5gyr.sfsanchored"
#run="rSFH_abias0.99_"$tscale"gyr.sfsflex"
#run="rSFH_0.2sfs_"$tscale"gyr.sfsflex"
#run="nodutycycle.sfsflex"
#run="rSFH_abias0.99_"$tscale"gyr.sfsflex"

#run="nodutycycle.sfsmf.sfsbroken"
#echo $run
#mpirun -np $NPROCS python /home/users/hahn/projects/centralMS/run/modelabc.py $run $niter > "/home/users/hahn/projects/centralMS/run/"$run"."$niter".modelabc.log"

for tscale in "10"; do #"0.5" "1" "2" "5"; do # "10"; do 
    #run="randomSFH"$tscale"gyr.sfsmf.sfsbroken"
    run="rSFH_abias0.5_"$tscale"gyr.sfsmf.sfsbroken"
    #run="rSFH_abias0.99_"$tscale"gyr.sfsmf.sfsbroken"
    echo $run
    mpirun -np $NPROCS python /home/users/hahn/projects/centralMS/run/modelabc.py $run $niter > "/home/users/hahn/projects/centralMS/run/"$run"."$niter".modelabc.log"
done 
