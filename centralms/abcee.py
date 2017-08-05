'''

use ABC-PMC 

'''
import os 
import numpy as np
import abcpmc
from abcpmc import mpi_util

# -- local -- 
import observables as Obvs
import models
import util as UT
import catalog as Cat

# --- plotting --- 
import matplotlib.pyplot as plt 
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


def Prior(run, shape='tophat'): 
    ''' Generate ABCPMC prior object given prior name, summary statistics, 
    and prior distribution shape. 

    Parameters
    ----------
    run : (string)
        String that specifies the ABC run  

    shape : (string)
        Specifies the shape of the prior. Default is tophat. Mainly only 
        tophat will be used...
    '''
    if shape != 'tophat': 
        raise NotImpelementError

    if run in ['test0', 'randomSFH']: 
        # SFMS_zslope, SFMS_mslope
        prior_min = [0.8, 0.4]
        prior_max = [1.2, 0.6]

    else:
        raise NotImplementedError

    prior_obj = abcpmc.TophatPrior(prior_min, prior_max) 

    return prior_obj


def SumData(sumstat, **data_kwargs): 
    ''' Return the summary statistics of data 
    '''
    sums = [] 
    if 'smf' in sumstat: # stellar mass function 
        if 'nsnap0' not in data_kwargs.keys():
            raise ValueError
        
        subhist = Cat.PureCentralHistory(nsnap_ancestor=data_kwargs['nsnap0'])
        subcat = subhist.Read(downsampled=None) # full sample

        smf = Obvs.getMF(subcat['m.star']) 

        sum = smf[1] # phi 
        sums.append(sum) 
    else: 
        raise NotImplementedError

    return sums


def SumSim(sumstat, subcat): #, **sim_kwargs): 
    ''' Return summary statistic of the simulation 
    
    parameters
    ----------
    sumstat : (list) 
        list of summary statistics to be included

    subcat : (obj)
        subhalo catalog output from models.model function 
    '''
    sums = [] 
    for stat in sumstat: 
        if stat == 'smf': # stellar mass function 

            # combine integrated stellar masses of SF galaxies
            # with SHAM stellar masses of the rest 
            # in principle we could compare to the SHAM MF * fQ...
            isSF = np.where(subcat['gclass'] == 'star-forming')
            isnotSF = np.where(subcat['gclass'] != 'star-forming')

            m_all = np.concatenate([subcat['m.star'][isSF], subcat['m.sham'][isnotSF]])
            w_all = np.concatenate([subcat['weights'][isSF], subcat['weights'][isnotSF]]) 

            smf = Obvs.getMF(m_all, weights=w_all)
            sum = smf[1] # phi 
        else: 
            raise NotImplementedError
        
        sums.append(sum) 

    return sums


def roe_wrap(sumstat, type='L2'):
    ''' Get it? Wrapper for Rhos or roe wrap. 
    '''
    if len(sumstat) == 1: # only SMF 
        if type == 'L2': 
            return L2_logSMF
    else: 
        raise NotImplementedError


def runABC(run, T, eps0, N_p=1000, sumstat=None, **run_kwargs): 
    ''' Main code for running ABC 

    Parameters
    ----------
    T : (int) 
        Number of iterations 

    eps0 : (list) 
        List of starting epsilon thresholds

    N_p : (int)
        Number of particles. Default is a 1000
    
    sumstat : (list) 
        List of strings that specifies which summary statistics
        will be used to calculate the distance metric. Options 
        include: 

    prior : (string)
        String that specifies the priors

    '''
    # check inputs 
    if len(eps0) != len(sumstat): 
        raise ValueError('Epsilon thresholds should correspond to number of summary statistics')
    
    # prior object 
    prior_obj = Prior(run, shape='tophat')

    # summary statistics of data 
    data_kwargs = {} 
    data_kwargs['nsnap0'] = run_kwargs['nsnap0']

    data_sum = SumData(sumstat, **data_kwargs)

    # summary statistics of simulation 
    sim_kwargs = {} 
    sim_kwargs['nsnap0'] = run_kwargs['nsnap0']
    sim_kwargs['downsampled'] = run_kwargs['downsampled']

    def Sim(tt): 
        sh_catalog = models.model(run, tt, **sim_kwargs)
        sums = SumSim(sumstat, sh_catalog)
        return sums 

    # distance metric 
    roe = roe_wrap(sumstat, type='L2')

    init_pool = None 
    # implement restart here
    # implement restart here

    # threshold 
    eps = abcpmc.ConstEps(T, eps0)
    try:
        mpi_pool = mpi_util.MpiPool()
        abcpmc_sampler = abcpmc.Sampler(
                N=N_p,                # N_particles
                Y=data_sum,             # data
                postfn=Sim,             # simulator 
                dist=roe,               # distance function  
                pool=mpi_pool)  

    except AttributeError: 
        abcpmc_sampler = abcpmc.Sampler(
                N=N_p,                # N_particles
                Y=data_sum,             # data
                postfn=Sim,            # simulator 
                dist=roe)           # distance function  

    # Write out all details of the run info 
    write_kwargs = {} 
    write_kwargs['Niter'] = T
    write_kwargs['sumstat'] = sumstat 
    for key in run_kwargs.keys():
        write_kwargs[key] = run_kwargs[key]
    Writeout('init', run, abcpmc_sampler, **write_kwargs)

    # particle proposal 
    abcpmc_sampler.particle_proposal_cls = abcpmc.ParticleProposal

    pools = []
    if init_pool is None:   # initiate epsilon write out
        Writeout('eps', run, None)

    print '----------------------------------------'
    for pool in abcpmc_sampler.sample(prior_obj, eps, pool=init_pool):
        print("T:{0},ratio: {1:>.4f}".format(pool.t, pool.ratio))
        print 'eps ', eps(pool.t)
        Writeout('eps', run, pool)

        # write out theta, weights, and distances to file 
        Writeout('theta', run, pool) 
        Writeout('w', run, pool) 
        Writeout('rho', run, pool) 
            
        # update epsilon based on median thresholding 
        if len(eps0) > 1: 
            eps.eps = np.median(np.atleast_2d(pool.dists), axis = 0)
        else: 
            eps.eps = np.median(pool.dists)
        print '----------------------------------------'
        pools.append(pool)

    return pools 


# different distance metric calculations 
def L2_logSMF(simsum, datsum): 
    ''' Measure the L2 norm for the case where the summary statistic 
    is the log(SMF). 
    '''
    if len(simsum[0]) != len(datsum[0]): 
        raise ValueError

    nonzero = np.where((simsum[0] > 0.) & (datsum[0] > 0.)) # preventing nans  
    n_bins = len(nonzero[0]) # number of bins 

    return np.sum((np.log10(simsum[0][nonzero]) - np.log10(datsum[0][nonzero]))**2)/np.float(n_bins)


def Writeout(type, run, pool, **kwargs): 
    ''' Given abcpmc pool object. Writeout specified ABC pool property
    '''
    file = UT.dat_dir()+'abc/'+run+'/'

    if type == 'init': # initialize
        if not os.path.exists(file): # make directory if it doesn't exist 
            os.makedirs(file)
        
        # write specific info of the run  
        file += 'info.md'
        f = open(file, 'w')
        f.write('# '+run+' run specs \n')
        f.write(''.join(['N_iter = ', str(kwargs['Niter']), '\n']))
        f.write(''.join(['N_particles = ', str(pool.N), '\n']))
        f.write(''.join(['Distance function = ', pool.dist.__name__ , '\n']))
        f.write('\n') 

        f.write(''.join(['Initial Snapshot = ', str(kwargs['nsnap0']), '\n']))
        f.write(''.join(['Downsampled by = ', str(kwargs['downsampled']), '\n']))
        f.close()

    elif type == 'eps': # threshold writeout 
        file += ''.join(['epsilon.', run, '.dat'])
        if pool is None: # write or overwrite threshold writeout
            f = open(file, "w")
        else: 
            f = open(file, "a") #append 
            f.write(str(pool.eps)+'\t'+str(pool.ratio)+'\n')
        f.close()
    elif type == 'theta': # particle thetas
        file += ''.join(['theta.t', str(pool.t), '.', run, '.dat']) 
        np.savetxt(file, pool.thetas) 
    elif type == 'w': # particle weights
        file += ''.join(['w.t', str(pool.t), '.', run, '.dat']) 
        np.savetxt(file, pool.ws)
    elif type == 'rho': # distance
        file += ''.join(['rho.t', str(pool.t), '.', run, '.dat']) 
        np.savetxt(file, pool.dists)
    else: 
        raise ValueError
