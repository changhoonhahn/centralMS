'''

use ABC-PMC 

'''
import numpy as np
import abcpmc
from abcpmc import mpi_util

# -- local -- 
import observables as Obvs
import models

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

    if run == 'test0': 
        # SFMS_zslope, SFMS_mslope
        prior_min = [0.8, 0.4]
        prior_max = [1.2, 0.6]

    else:
        raise NotImplementedError

    prior = abcpmc.TophatPrior(prior_min, prior_max) 

    return prior_obj


def SumData(sumstat, **data_kwargs): 
    ''' Return the summary statistics of data 
    '''
    sums = [] 
    for stat in sumstat: 
        if stat == 'smf': # stellar mass function 
            if 'nsnap0' not in data_kwargs.keys():
                raise ValueError
            
            subhist = Cat.PureCentralHistory(nsnap_ancestor=data_kwargs['nsnap0'])
            subcat = subhist.Read(downsampled=None) # full sample

            smf = Obvs.getMF(subcat['m.sham']) 

            sum = smf[0] # phi 
        else: 
            raise NotImplementedError
        
        sums.append(sum) 

    return sums


def SumSim(sumstat, subcat, **sim_kwargs): 
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
            #####
            #####
            #####
            #####
            #####
            #####

            sum = smf[0] # phi 
        else: 
            raise NotImplementedError
        
        sums.append(sum) 

    return sums


def roe_wrap(sum_stat):
    ''' Get it? Wrapper for Rhos or roe wrap. 
    '''
    if sum_stat 



def run_ABC(T, eps0, run, N_p=1000, sumstat=None, prior=None, **run_kwargs): 
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
        sums = SumSim(sumstat, subcat)
        return sums 

    # distance metric 
    roe = roe_wrap(sumstat)
