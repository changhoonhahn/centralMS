'''
'''
import numpy as np 
# --- centralms --- 
from centralMS import util as UT
from centralMS import abcee as ABC
from centralMS import observables as Obvs 

def minimize(tduty): 
    ''' 
    '''
    if tduty not in [0.5, 1, 2, 5, 10]: 
        raise ValueError 
    run = ''.join(['randomSFH_', str(tduty), 'gyr']) 
    ABC.minimize(run, sumstat=['smf'], nsnap0=15, downsampled='20') 
    return None 


def noAbiasABC(tduty, Niter=14, Npart=1000): 
    ''' ABC run without assembly bias 
    '''
    if tduty not in [0.5, 1, 2, 5, 10]: 
        raise ValueError 
    run = ''.join(['randomSFH_', str(tduty), 'gyr']) 
    prior = ABC.Prior('anchored', shape='tophat') 
    ABC.runABC(run, Niter, [1.e5], prior, N_p=Npart, sum_stat=['smf'], nsnap0=15, downsampled='20') 
    return None 


if __name__=="__main__":
    minimize(10) 
    #noAbiasABC(10, Niter=2, Npart=10) # test 
