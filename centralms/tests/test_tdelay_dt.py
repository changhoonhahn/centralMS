import sys
import os 
import numpy as np 

# -- local -- 
import env 
import abcee
import util as UT
import observables as Obvs
import corner as DFM 

import emcee
from emcee.utils import MPIPool

# --- plotting --- 
import matplotlib.pyplot as plt 
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


def tdelay_dt_mcmc(run, theta, Niter=20, Nwalkers=10, Ndim=2, sigma_smhm=0.2, nsnap0=15, downsampled='14', flag=None, continue_chain=False): 
    '''
    '''
    if Ndim == 2: 
        tdelay_range = [0., 3.]#np.arange(0., 3., 0.5)
        dt_range = [0.1, 4.]

    # new chain 
    chain_file = ''.join([UT.fig_dir(), run, '.tdelay_dt_mcmc.chain.dat']) 
    if os.path.isfile(chain_file) and continue_chain:   
        print 'Continuing previous MCMC chain!'
        sample = np.loadtxt(chain_file) 
        Niter = Niter - (np.float(len(sample))/np.float(Nwalkers)) # Number of chains left to finish 
        if Niter <= 0: 
            raise ValueError
            print Niter, ' iterations left to finish'
    else: 
        f = open(chain_file, 'w')
        f.close()
        # Initializing Walkers
        pos0 = [np.array([np.random.uniform(tdelay_range[0], tdelay_range[1]), np.random.uniform(dt_range[0], dt_range[1])]) for i in range(Nwalkers)]

    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    # Initializing the emcee sampler
    kwargs = {
            'theta': theta, 
            'sigma_smhm': 0.2, 
            'nsnap0': 15, 
            'downsampled': '14', 
            }
    sampler = emcee.EnsembleSampler(Nwalkers, Ndim, sigM, pool=pool, kwargs=kwargs)
    for result in sampler.sample(pos0, iterations=Niter, storechain=False):
        position = result[0]
        #print position
        f = open(chain_file, 'a')
        for k in range(position.shape[0]): 
            output_str = '\t'.join(position[k].astype('str')) + '\n'
            f.write(output_str)
        f.close()
    pool.close()

    return None 

    
def sigM(tt, theta=np.array([1.35, 0.6]), sigma_smhm=0.2, nsnap0=15, downsampled='14'):
    smhmr = Obvs.Smhmr()
    theta_i = np.concatenate([theta, np.array([tt[0], tt[1]])])
    try: 
        subcat_sim = abcee.model(run, theta_i, 
                nsnap0=nsnap0, sigma_smhm=sigma_smhm, downsampled=downsampled) 
        sumsim = abcee.SumSim(['smf'], subcat_sim, info=True)

        isSF = np.where(subcat_sim['gclass'] == 'sf') # only SF galaxies 
        #calculate sigma_M* at M_h = 12
        m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_sim['halo.m'][isSF], subcat_sim['m.star'][isSF], 
                dmhalo=0.2, weights=subcat_sim['weights'][isSF])
        return sig_mhalo[np.argmin(np.abs(m_mid-12.))]
    except ValueError: 
        return 


if __name__=='__main__': 
    run = 'rSFH_r0.99_delay_dt_test'
    theta = np.array([1.35, 0.6])
    Niter = int(sys.argv[1])
    Nwalkers = int(sys.argv[2])

    tdelay_dt_mcmc(run, theta, Niter=Niter, Nwalkers=Nwalkers, Ndim=2,
            sigma_smhm=0.2, nsnap0=15, downsampled='14')
