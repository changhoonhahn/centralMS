'''

use ABC-PMC 

'''
import numpy as np
import abcpmc
from abcpmc import mpi_util

# -- local -- 
import observables as Obvs

# --- plotting --- 
import matplotlib.pyplot as plt 
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors

def BlankTheta(run, *args): 
    ''' Given the ABC run, specify the blank variables 
    '''
    theta_fixed = {} 
    if run == 'test0': 

        theta_fixed['gv'] = {'slope': 1.03, 'fidmass': 10.5, 'offset': -0.02}
        theta_fixed['fq'] = {'name': 'cosmos_tinker'}
        theta_fixed['fpq'] = {'slope': -2.079703, 'offset': 1.6153725, 'fidmass': 10.5}

        theta_fixed['sfms'] = {'name': 'linear', 'zslope': 1.05, 'mslope':0.53}
        theta_fixed['mass'] = {'solver': 'euler', 'f_retain': 0.6, 't_step': 0.05} 
        theta_fixed['sfh'] = {'name': 'constant_offset'}
        theta_fixed['sfh']['nsnap0'] = 15 
    else: 
        raise NotImplementedError


def Prior(prior, sumstat, shape='tophat'): 
    ''' Generate ABCPMC prior object given prior name, summary statistics, 
    and prior distribution shape. 

    Parameters
    ----------
    prior : (string)
        String that specifies the priors

    sumstat : (list) 
        List of strings that specifies which summary statistics
        will be used to calculate the distance metric. Options 
        include: 

    shape : (string)
        Specifies the shape of the prior. Default is tophat. Mainly only 
        tophat will be used...
    '''
    if shape != 'tophat': 
        raise NotImpelementError

    if prior == 'testing': 
        prior_min, prior_max = [], [] 

        for stat in sumstat: 
            if 'smf' 

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


def run_ABC(T, eps0, N_p=1000, sumstat=None, prior=None, run=None, **run_kwargs): 
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
    prior_obj = Prior(prior, run, shape='tophat')

    # summary statistics of data 
    data_kwargs = {} 
    data_kwargs['nsnap0'] = run_kwargs['nsnap0']

    data_sum = SumData(sumstat, **data_kwargs)


    # summary statistics of simulation 


    # distance metric 

