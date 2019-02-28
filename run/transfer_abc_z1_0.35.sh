# !/bin/bash/
rcorr=$1
tduty=$2

if (( $(echo "$rcorr > 0." |bc -l) )); then 
    run="rSFH_abias"$rcorr"_"$tduty"gyr.sfsmf.sigma_z1_0.35.sfsbroken"
else
    run="randomSFH"$tduty"gyr.sfsmf.sigma_z1_0.35.sfsbroken"
fi 
remote="sirocco"
remote_dir="/mount/sirocco1/hahn/centralms/abc/"

local_dir=$CENTRALMS_DIR"abc/"
# make directory if it doesn't exist 
mkdir -p $local_dir$run"/"
mkdir -p $local_dir$run"/model/"

cmd=$remote":"$remote_dir$run"/{info.md"

t=14
cmd=$cmd",theta.t"$t"."$run".dat"
cmd=$cmd",w.t"$t"."$run".dat"
cmd=$cmd",rho.t"$t"."$run".dat"
cmd=$cmd"}"
scp $cmd $local_dir$run

cmd=$remote":"$remote_dir$run"/model/{model.theta_median0.t"$t".p"
for i in $(seq 1 9); do 
    cmd=$cmd",model.theta_median"$i".t"$t".p"
done 
for i in $(seq 0 99); do 
    cmd=$cmd",model.theta"$i".t"$t".p"
done 
cmd=$cmd"}"
scp $cmd $local_dir$run"/model/"
