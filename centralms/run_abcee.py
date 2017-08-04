'''
Simple wrapper for running ABC with commandline arguments 
'''
import sys 
from abcee import runABC

#restart = int(sys.argv[1])
#if restart == 0: 

# ABC run name 
print 'ABC run name = ', abcrun
abcrun = sys.argv[1]


# number of iterations 
Niter = int(sys.argv[2])
print 'N_iterations = ', Niter

# number of particles
Npart = int(sys.argv[3])
print 'N_particle = ', Npart

# number of summary statistics 
n_sum = int(sys.argv[4]) 
if n_sum == 0:       # SSFR only distance metric
    print 'SMF'
    sum_stat = ['smf'] 
else: 
    raise ValueError 
    
esp0 = [10. for i in range(n_sum)]

# plus some hardcoded kwargs
nsnap0=15 # starting snapshot 
downsampled='14' # downsample amount 

runABC(run, Niter, eps0, N_p=Npart, sumstat=sum_stat, nsnap0=nsap0, downsampled=downsampled) 

'''
elif restart == 1:  
    raise NotImplementedError('Reimplement... carefully.') Niter = int(sys.argv[2]) print 'N_iterations = ', Niter Npart = int(sys.argv[3]) print 'N_particle = ', Npart abcrun = sys.argv[4] print 'ABC run name = ', abcrun trestart = int(sys.argv[5]) print 'Restarting abc from t = ', str(trestart) ABC(Niter, [4.22460181, 0.33794247], Npart=Npart, prior_name='try0', observables=['ssfr', 'fqz03'], abcrun=abcrun, t_restart=trestart, restart=True)
                                                                                                                                                                                                                                                            '''
