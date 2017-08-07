'''

use ABC-PMC 

'''
import os 
import numpy as np
import abcpmc
from abcpmc import mpi_util
import corner as DFM

# -- local -- 
import observables as Obvs
import util as UT
import catalog as Cat
import evolver as Evol

import matplotlib.pyplot as plt 

def Theta(run): 
    tt = {} 
    if run in ['test0']: 
        tt['variable'] = ['SFMS z slope', 'SFMS m slope']
        tt['label'] = ['$\mathtt{m_{z; SFMS}}$', '$\mathtt{m_{M_*; SFMS}}$']
    
    return tt


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
        prior_max = [1.3, 0.6]
    else:
        raise NotImplementedError

    prior_obj = abcpmc.TophatPrior(prior_min, prior_max) 

    return prior_obj


def Data(**data_kwargs): 
    ''' Our 'data'
    '''
    subhist = Cat.PureCentralHistory(nsnap_ancestor=data_kwargs['nsnap0'])
    subcat = subhist.Read(downsampled=None) # full sample
    return subcat


def SumData(sumstat, info=False, **data_kwargs): 
    ''' Return the summary statistics of data 
    '''
    subcat = Data(**data_kwargs)

    sums = [] 
    if 'smf' in sumstat: # stellar mass function 
        if 'nsnap0' not in data_kwargs.keys():
            raise ValueError

        marr_fixed = np.arange(9., 12.2, 0.2)
        smf = Obvs.getMF(subcat['m.star'], m_arr=marr_fixed) 

        if not info: 
            sum = smf[1] # phi 
        else: 
            sum = smf # mass, phi 

        sums.append(sum) 
    else: 
        raise NotImplementedError

    return sums


def model(run, args, **kwargs): 
    ''' model given the ABC run 
    '''
    theta = {}

    if run in ['test0', 'randomSFH']: 
        # args = SFMS_zslope, SFMS_mslope

        # these values were set by cenque project's output
        theta['gv'] = {'slope': 1.03, 'fidmass': 10.5, 'offset': -0.02}
        theta['fq'] = {'name': 'cosmos_tinker'}
        theta['fpq'] = {'slope': -2.079703, 'offset': 1.6153725, 'fidmass': 10.5}
        
        # for simple test 
        theta['mass'] = {'solver': 'euler', 'f_retain': 0.6, 't_step': 0.05} 
        theta['sfh'] = {'name': 'constant_offset'}
        theta['sfh']['nsnap0'] =  kwargs['nsnap0'] 
            
        # SFMS slopes can change 
        theta['sfms'] = {'name': 'linear', 'zslope': args[0], 'mslope': args[1]}

        # load in Subhalo Catalog (pure centrals)
        subhist = Cat.PureCentralHistory(nsnap_ancestor=kwargs['nsnap0'])
        subcat = subhist.Read(downsampled=kwargs['downsampled']) # full sample

        eev = Evol.Evolver(subcat, theta, nsnap0=kwargs['nsnap0'])
        eev.Initiate()
        eev.Evolve() 
    else: 
        raise NotImplementedError

    return eev.SH_catalog


def SumSim(sumstat, subcat, info=False): #, **sim_kwargs): 
    ''' Return summary statistic of the simulation 
    
    parameters
    ----------
    sumstat : (list) 
        list of summary statistics to be included

    subcat : (obj)
        subhalo catalog output from model function 

    info : (bool)
        specify extra info. Default is 0 
    '''
    sums = [] 
    for stat in sumstat: 
        if stat == 'smf': # stellar mass function 
            marr_fixed = np.arange(9., 12.2, 0.2)
            smf = Obvs.getMF(subcat['m.star'], weights=subcat['weights'], m_arr=marr_fixed)
            if not info: 
                sum = smf[1] # phi 
            else: 
                sum = smf # m_bin, phi 

            # combine integrated stellar masses of SF galaxies
            # with SHAM stellar masses of the rest 
            # in principle we could compare to the SHAM MF * fQ...
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
        sh_catalog = model(run, tt, **sim_kwargs)
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


def readABC(run, T): 
    ''' Read in theta, w, and rho from ABC writeouts
    '''
    dir = UT.dat_dir()+'abc/'+run+'/'
    
    file = lambda ss, t, r: ''.join([dir, ss, '.t', str(t), '.', r, '.dat'])
    
    abc_out = {} 
    # read in theta, w, rho 
    abc_out['theta'] = np.loadtxt(file('theta', T, run)) 
    abc_out['w'] = np.loadtxt(file('w', T, run))
    abc_out['rho'] = np.loadtxt(file('rho', T, run))

    return abc_out


def plotABC(run, T): 
    ''' Corner plots of ABC runs  
    '''
    # thetas
    abcout = readABC(run, T) 
    theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) for i in range(len(abcout['theta'][0]))]

    theta_info = Theta(run) 
    
    # prior
    prior_obj = Prior(run)
    prior_range = [(prior_obj.min[i], prior_obj.max[i]) for i in range(len(prior_obj.min))]
    
    # figure name 
    fig_name = ''.join([UT.dat_dir(), 'abc/', run, '/', 't', str(T), '.', run , '.png'])
    
    fig = DFM.corner(abcout['theta'], weights=abcout['w'].flatten(),
            truths=theta_med, truth_color='#ee6a50', # median theta 
            labels=theta_info['label'], label_kwargs={'fontsize': 25},
            range=prior_range,
            quantiles=[0.16,0.5,0.84],
            show_titles=True,
            title_args={"fontsize": 12},
            plot_datapoints=True,
            fill_contours=True,
            levels=[0.68, 0.95], 
            color='#ee6a50', 
            bins=20, 
            smooth=1.0)
    plt.savefig(fig_name) 
    plt.close()
    return None 
