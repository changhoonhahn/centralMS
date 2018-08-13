# !/bin/bash/

for tscale in "10"; do # "0.5" "1" "2" "5" "10"; do 
    echo "transfering model(ABC partciles) for ..."$tscale 
    sh /Users/chang/projects/centralMS/run/transfer_modelabc.sh "rSFH_abias0.99_"$tscale"gyr.sfsflex" 14
done
