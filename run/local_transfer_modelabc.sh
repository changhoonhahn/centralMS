# !/bin/bash/

for tscale in "5"; do # "10"; do 
    echo "transfering model(ABC partciles) for ..."$tscale 
    #sh /Users/chang/projects/centralMS/run/transfer_modelabc.sh "randomSFH"$tscale"gyr.sfsmf.sfsbroken" 14
    sh /Users/chang/projects/centralMS/run/transfer_modelabc.sh "rSFH_abias0.5_"$tscale"gyr.sfsmf.sfsbroken" 14
    sh /Users/chang/projects/centralMS/run/transfer_modelabc.sh "rSFH_abias0.99_"$tscale"gyr.sfsmf.sfsbroken" 14
    #sh /Users/chang/projects/centralMS/run/transfer_modelabc.sh "rSFH_abias0.99_"$tscale"gyr.sfsflex" 14
done
