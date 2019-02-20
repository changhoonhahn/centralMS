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
    run = ''.join(['randomSFH', tduty, 'gyr.sfsmf.sfs', sfs]) 
    print(run)
    prior = ABC.Prior(sfs, shape='tophat') 
    ABC.runABC(run, Niter, [1.e5], prior, N_p=Npart, sumstat=['sfsmf'], nsnap0=15, downsampled='20') 
    return None 


def AbiasABC(tduty, rcorr=0.5, sfs='flex', Niter=14, Npart=1000): 
    ''' ABC run without assembly bias 
    '''
    if tduty not in ['0.5', '1', '2', '5', '10']: 
        raise ValueError 
    run = ''.join(['rSFH_abias', str(rcorr), '_', tduty, 'gyr.sfsmf.sfs', sfs]) 
    print(run)
    prior = ABC.Prior(sfs, shape='tophat') 
    ABC.runABC(run, Niter, [1.e5], prior, N_p=Npart, sumstat=['sfsmf'], nsnap0=15, downsampled='20') 
    return None 


def AbiasABC_z1sigma(tduty, rcorr=0.5, sfs='flex', sigma_z1=0.35, Niter=14, Npart=1000): 
    ''' ABC run with assembly bias specified by correlation coefficient r with 
    sigma_logM* at z_init given by sigma_z1. 
    '''
    if tduty not in ['0.5', '1', '2', '5', '10']: 
        raise ValueError 
    run = ''.join(['rSFH_abias', str(rcorr), '_', tduty, 'gyr.sfsmf.sfs', sfs, '.sigma_z1_%.2f' % sigma_z1]) 
    print(run)
    prior = ABC.Prior(sfs, shape='tophat') 
    ABC.runABC(run, Niter, [1.e5], prior, N_p=Npart, sumstat=['sfsmf'], nsnap0=15, sigma_smhm=sigma_z1, downsampled='20') 
    return None 


def narrowSFS_noAbiasABC(tduty, sfs='flex', Niter=14, Npart=1000): 
    ''' ABC run without assembly bias 
    '''
    if tduty not in ['0.5', '1', '2', '5', '10']: 
        raise ValueError 
    run = ''.join(['rSFH_0.2sfs_', tduty, 'gyr.sfs', sfs]) 
    print(run)
    prior = ABC.Prior(sfs, shape='tophat') 
    ABC.runABC(run, Niter, [1.e5], prior, N_p=Npart, sumstat=['smf'], nsnap0=15, downsampled='20') 
    return None 


def nodutycycle(sfs='flex', Niter=14, Npart=1000): 
    ''' ABC run without assembly bias 
    '''
    run = ''.join(['nodutycycle.sfsmf.sfs', sfs]) 
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
        rcorr = float(sys.argv[2])
        tduty = sys.argv[3]
        sfs = sys.argv[4]
        niter = int(sys.argv[5]) 
        npart = int(sys.argv[6]) 
        AbiasABC(tduty, rcorr=rcorr, sfs=sfs, Niter=niter, Npart=npart) # test 
    elif name == 'abias_z1sigma': # tduty, assembly bias, and sigma_logM* at zinit 
        rcorr = float(sys.argv[2])
        tduty = sys.argv[3]
        sfs = sys.argv[4]
        sigmaz1 = sys.argv[5]
        niter = int(sys.argv[6]) 
        npart = int(sys.argv[7]) 
        AbiasABC_z1sigma(tduty, rcorr=rcorr, sfs=sfs, sigma_z1=sigmaz1, Niter=niter, Npart=npart)
    elif name == 'narrow_noabias': 
        tduty = sys.argv[2]
        sfs = sys.argv[3]
        niter = int(sys.argv[4]) 
        npart = int(sys.argv[5]) 
        narrowSFS_noAbiasABC(tduty, sfs=sfs, Niter=niter, Npart=npart) # test 
    elif name == 'nodutycycle': 
        sfs = sys.argv[2]
        niter = int(sys.argv[3]) 
        npart = int(sys.argv[4]) 
        nodutycycle(sfs=sfs, Niter=niter, Npart=npart) # test 
    elif name == 'modelrun': 
        run = sys.argv[2]
        niter = int(sys.argv[3]) 
        modelABCpool(run, niter)
    else: 
        raise NotImplementedError
