#!/bin/bash/
# bash script for running ABC on local machine 

# constant offset SFH (ABC run = test0) 
#python /Users/chang/projects/centralMS/centralms/run/run_abcee.py test0 2 10 1 0 

# random fluctuating SFH (ABC run = randomSFH_5gyr) 
#python /Users/chang/projects/centralMS/centralms/run/run_abcee.py randomSFH_5gyr 2 10 1 0 

# narrow SFMS test: random fluctuating SFH (ABC run = randomSFH_5gyr_narrSFMS) 
#python /Users/chang/projects/centralMS/centralms/run/run_abcee.py randomSFH_5gyr_narrSFMS 2 10 1 0 

# random fluctuating SFH correlated (r = 0.99) with halo growth over tdyn = 2.5 Gyr and 
# dutycycle of 0.5 Gyr
# ABC run = rSFH_r0.99_tdyn_0.5Gyr
python /Users/chang/projects/centralMS/centralms/run/run_abcee.py rSFH_r0.99_tdyn_0.5Gyr 2 10 1 0 
