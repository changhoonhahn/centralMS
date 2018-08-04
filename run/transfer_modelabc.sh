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
mkdir -p $local_dir$run"/model/"

cmd=$remote":"$remote_dir$run"/model/{model.theta_median0.t"$2".p"

for i in $(seq 1 9); do 
    cmd=$cmd",model.theta_median"$i".t"$2".p"
done 

for i in $(seq 0 999); do 
    cmd=$cmd",model.theta"$i".t"$2".p"
done 
cmd=$cmd"}"
#echo $cmd
scp $cmd $local_dir$run"/model/"
