# !/bin/bash/
run="randomSFH_5gyr"
remote="sirocco"
if [ $remote = "sirocco" ]; then 
    remote_dir="/mount/sirocco1/hahn/centralms/abc/"
elif [ $remote = "harmattan" ]; then
    remote_dir="/data1/hahn/centralMS/"
fi 

local_dir=$CENTRALMS_DIR"abc/"

cmd=$remote":"$remote_dir$run"/{info.md"

for t in {0..6}; do 
    cmd=$cmd",theta.t"$t"."$run".dat"
    cmd=$cmd",w.t"$t"."$run".dat"
    cmd=$cmd",rho.t"$t"."$run".dat"
done 
cmd=$cmd"}"
echo $cmd
scp $cmd $local_dir$run
