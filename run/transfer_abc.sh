# !/bin/bash/
run=$1
remote="sirocco"
if [ $remote = "sirocco" ]; then 
    remote_dir="/mount/sirocco1/hahn/centralms/abc/"
elif [ $remote = "harmattan" ]; then
    remote_dir="/data1/hahn/centralMS/"
fi 

local_dir=$CENTRALMS_DIR"abc/"
# make directory if it doesn't exist 
mkdir -p $local_dir$run"/"

cmd=$remote":"$remote_dir$run"/{info.md"

for t in $(seq 0 $2); do #{0..$2}; do 
    cmd=$cmd",theta.t"$t"."$run".dat"
    cmd=$cmd",w.t"$t"."$run".dat"
    cmd=$cmd",rho.t"$t"."$run".dat"
done 
cmd=$cmd"}"
#echo $cmd
scp $cmd $local_dir$run
