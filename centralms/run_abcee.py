'''
Simple wrapper for running ABC with commandline arguments 
'''
import sys 
from abcee import runABC

#restart = int(sys.argv[1])
#if restart == 0: 

# ABC run name 
abcrun = sys.argv[1]
print 'Run ', abcrun

# number of iterations 
Niter = int(sys.argv[2])
print 'N_iterations = ', Niter

# number of particles
Npart = int(sys.argv[3])
print 'N_particle = ', Npart

# number of summary statistics 
n_sum = int(sys.argv[4]) 
if n_sum == 1:       # SSFR only distance metric
    print 'SMF'
    sum_stat = ['smf'] 
else: 
    raise ValueError 
    
eps0 = [1.e5 for i in range(n_sum)]

# plus some hardcoded kwargs
nsnap0=15 # starting snapshot 
downsampled='14' # downsample amount 

notify = int(sys.argv[5])
if notify == 1: 
    runABC(abcrun, Niter, eps0, N_p=Npart, sumstat=sum_stat, notify=True, nsnap0=nsnap0, downsampled=downsampled) 
else: 
    print 'No notification issued'
    runABC(abcrun, Niter, eps0, N_p=Npart, sumstat=sum_stat, notify=False, nsnap0=nsnap0, downsampled=downsampled) 
