'''
'''
import sys 
import numpy as np 
# --- centralms --- 
from centralms import util as UT
from centralms import abcee as ABC
from centralms import observables as Obvs 

def minimize(tduty): 
    ''' 
    '''
    if tduty not in [0.5, 1, 2, 5, 10]: 
        raise ValueError 
    run = ''.join(['randomSFH_', str(tduty), 'gyr']) 
    ABC.minimize(run, sumstat=['smf'], nsnap0=15, downsampled='20') 
    return None 


def noAbiasABC(tduty, sfs='flex', Niter=14, Npart=1000): 
    ''' ABC run without assembly bias 
    '''
    if tduty not in ['0.5', '1', '2', '5', '10']: 
        raise ValueError 
    run = ''.join(['randomSFH', tduty, 'gyr.sfs', sfs]) 
    print(run)
    prior = ABC.Prior(sfs, shape='tophat') 
    ABC.runABC(run, Niter, [1.e5], prior, N_p=Npart, sumstat=['smf'], nsnap0=15, downsampled='20') 
    return None 


def modelABCpool(run, t): 
    ''' evaluate model(theta) for all theta in ABC pool 
    '''
    ABC.model_ABCparticle(run, t, nsnap0=15, downsampled='20') 
    return None 


if __name__=="__main__":
    name = sys.argv[1]
    if name == 'noabias': 
        tduty = sys.argv[2]
        sfs = sys.argv[3]
        niter = int(sys.argv[4]) 
        npart = int(sys.argv[5]) 
        noAbiasABC(tduty, sfs=sfs, Niter=niter, Npart=npart) # test 
    elif name == 'abias': 
        raise ValueError
    elif name == 'modelrun': 
        run = sys.argv[2]
        niter = int(sys.argv[3]) 
        modelABCpool(run, niter)
    else: 
        raise NotImplementedError
