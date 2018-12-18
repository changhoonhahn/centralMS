# !/bin/bash/
for tscale in "0.5" "1" "2" "5"; do # "10"; do 
    echo "transfering ABC partciles for ..."$tscale 
    sh /Users/chang/projects/centralMS/run/transfer_abc.sh "rSFH_abias0.99_"$tscale"gyr.sfsmf.sfsbroken" 14
    #sh /Users/chang/projects/centralMS/run/transfer_abc.sh "rSFH_abias0.5_"$tscale"gyr.sfsmf.sfsbroken" 14
    #sh /Users/chang/projects/centralMS/run/transfer_abc.sh "randomSFH"$tscale"gyr.sfsmf.sfsbroken" 14
    #sh /Users/chang/projects/centralMS/run/transfer_abc.sh "randomSFH"$tscale"gyr.sfsmf.sfsflex" 14
    #sh /Users/chang/projects/centralMS/run/transfer_abc.sh "rSFH_abias0.99_"$tscale"gyr.sfsflex" 14
done
