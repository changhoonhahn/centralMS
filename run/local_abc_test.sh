# !/bin/bash

# No Assembly Bias; Flex SFS
tduty=5
run="randomSFH"$tduty"gyr.sfsflex"
python /Users/chang/projects/centralMS/run/abc.py noabias $tduty 2 10 > /Users/chang/projects/centralMS/run/local_$run.log
