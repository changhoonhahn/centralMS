'''

use ABC-PMC 

'''
import os 
import h5py 
import time
import numpy as np
import scipy as sp 
import abcpmc
from abcpmc import mpi_util
# -- local -- 
from . import util as UT
from . import sfh as SFH
from . import catalog as Cat
from . import evolver as Evol
from . import observables as Obvs


def Theta(): 
    tt = {} 
    #tt['variable'] = ['SFMS z slope', 'SFMS m slope']#, 'SFMS offset']
    #tt['label'] = ['$m_{z; SFMS}$', '$m_{M_*; SFMS}$']#, '$c_\mathrm{SFMS}$']
    tt['variable'] = ['SFMS amp z param', 'SFMS slope z param']
    tt['label'] = ['$m_{z; amp}$', '$m_{z; slope}$']
    return tt


def Prior(name, shape='tophat'): 
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
    if name not in ['anchored', 'flex']: 
        raise ValueError
    
    if name == 'anchored': 
        # new priors since we implemented "anchored" SFMS  
        # SFMS amplitude z-dep parameter, SFMS slope z-dep parameter
        prior_min = [0.5, -0.5]#, -0.15]
        prior_max = [2.5, 0.5]#, -0.06]
    elif name == 'flex': 
        # SFMS_zslope, SFMS_mslope
        prior_min = [1., 0.0]#, -0.15]
        prior_max = [2., 0.8]#, -0.06]
    return  abcpmc.TophatPrior(prior_min, prior_max) 


def dataSum(sumstat=['smf']): 
    ''' Summary statistics of the data (SDSS) 
    '''
    sums = [] 
    if 'smf' in sumstat: # central stellar mass function 
        marr, smf, _ = Obvs.dataSMF(source='li-white') # Li & White SMF 
        fcen = (1. - np.array([Obvs.f_sat(mm, 0.05) for mm in marr])) # sallite fraction 
        sums.append(fcen * smf) 
    else: 
        raise NotImplementedError
    return sums


def model(run, args, **kwargs): 
    ''' model given the ABC run 
    '''
    theta = _model_theta(run, args) # return theta(run, args)  

    # load in Subhalo Catalog (pure centrals)
    if 'sigma_smhm' in kwargs.keys(): 
        censub = Cat.CentralSubhalos(nsnap0=kwargs['nsnap0'], 
                sigma_smhm=kwargs['sigma_smhm'])
    else: 
        censub = Cat.CentralSubhalos(nsnap0=kwargs['nsnap0'])
    shcat = censub.Read(downsampled=kwargs['downsampled']) 
    shcat = Evol.Evolve(shcat, theta) 
    return shcat 


def _model_theta(run, args): 
    ''' return theta given run and args 
    '''
    theta = {}
    # parameters for stellar mass integration  
    theta['mass'] = {'solver': 'euler', 'f_retain': 0.6, 't_step': 0.05} 
    # SFMS slopes can change 
    if run.split('.sfs')[-1] == 'flex': 
        theta['sfms'] = {'name': 'flex', 'zslope': args[0], 'mslope': args[1], 'sigma': 0.3}
    elif run.split('.sfs')[-1] == 'anchored': 
        theta['sfms'] = {'name': 'anchored', 'amp': args[0], 'slope': args[1], 'sigma': 0.3}
    
    if 'randomSFH' in run: 
        # run = randomSFH%fgyr.sfms
        # SFH that randomly fluctuates on a tduty timescale
        theta['sfh'] = {'name': 'random_step_fluct'} 
        tduty = float(run.split('randomSFH')[-1].split('gyr')[0]) 
        theta['sfh']['dt_min'] = tduty
        theta['sfh']['dt_max'] = tduty
        theta['sfh']['sigma'] = 0.3 
    else: 
        raise ValueError
    '''
        elif run == 'rSFH_r1.0_most': 
            theta['sfh'] = {'name': 'random_step_most_abias'}
            theta['sfh']['dt_min'] = 0.5 
            theta['sfh']['dt_max'] = 0.5 
            theta['sfh']['sigma_tot'] = 0.3 
            theta['sfh']['sigma_corr'] = 0.3
        elif run == 'rSFH_r0.66_delay': 
            theta['sfh'] = {'name': 'random_step_abias_delay_dz'}
            theta['sfh']['dt_min'] = 0.5 
            theta['sfh']['dt_max'] = 0.5 
            theta['sfh']['sigma_tot'] = 0.3 
            theta['sfh']['sigma_corr'] = 0.66 * 0.3
            theta['sfh']['dt_delay'] = 1. # Gyr 
            theta['sfh']['dz_dMh'] = 0.5 
        elif run == 'rSFH_r0.99_delay': 
            theta['sfh'] = {'name': 'random_step_abias_delay_dz'}
            theta['sfh']['dt_min'] = 0.5 
            theta['sfh']['dt_max'] = 0.5 
            theta['sfh']['sigma_tot'] = 0.3 
            theta['sfh']['sigma_corr'] = 0.99 * 0.3
            theta['sfh']['dt_delay'] = 1. # Gyr 
            theta['sfh']['dz_dMh'] = 0.5 
        elif run == 'rSFH_r0.99_tdyn_5gyr': 
            # random SFH with 0.99 correlation with halo growth 
            # over t_dyn and duty cycle of 5 Gyr 
            theta['sfh'] = {'name': 'random_step_abias_delay_dt'}
            theta['sfh']['dt_min'] = 5. 
            theta['sfh']['dt_max'] = 5. 
            theta['sfh']['sigma_tot'] = 0.3 
            theta['sfh']['sigma_corr'] = 0.99 * 0.3
            theta['sfh']['dt_delay'] = 0. # Gyr 
            theta['sfh']['dt_dMh'] = 2.5 # Gyr
        elif run == 'rSFH_r0.99_tdyn_0.5gyr_narrSFMS': 
            # random SFH with 0.99 correlation with halo growth 
            # over t_dyn and duty cycle of 0.5 Gyr 
            theta['sfh'] = {'name': 'random_step_abias_delay_dt'}
            theta['sfh']['dt_min'] = 0.5 
            theta['sfh']['dt_max'] = 0.5 
            theta['sfh']['sigma_tot'] = 0.2 # note narrower SFMS
            theta['sfh']['sigma_corr'] = 0.99 * 0.2 # note narrower SFMS
            theta['sfh']['dt_delay'] = 0. # Gyr 
            theta['sfh']['dt_dMh'] = 2.5 # Gyr
            theta['sfms']['sigma'] = 0.2 # note narrower SFMS
        elif run == 'rSFH_r0.99_delay_dt_test': 
            theta['sfh'] = {'name': 'random_step_abias_delay_dt'}
            theta['sfh']['dt_min'] = args[2]
            theta['sfh']['dt_max'] = args[2] 
            theta['sfh']['sigma_tot'] = 0.3 
            theta['sfh']['sigma_corr'] = 0.99 * 0.3
            theta['sfh']['dt_delay'] = args[3] # Gyr 
            theta['sfh']['dt_dMh'] = args[4]  # Gyr
        elif run == 'rSFH_r_delay_dt_test': 
            theta['sfh'] = {'name': 'random_step_abias_delay_dt'}
            theta['sfh']['dt_min'] = args[3]
            theta['sfh']['dt_max'] = args[3] 
            theta['sfh']['sigma_tot'] = 0.3 
            theta['sfh']['sigma_corr'] = args[2] * 0.3
            theta['sfh']['dt_delay'] = args[4] # Gyr 
            theta['sfh']['dt_dMh'] = args[5]  # Gyr
        else: 
            raise NotImplementedError
    '''
    return theta


def modelSum(cencat, sumstat=['smf']): 
    ''' Return summary statistic of the simulation 
    
    parameters
    ----------
    sumstat : (list) 
        list of summary statistics to be included

    cencat : (obj)
        central subhalo catalog output from model function 

    info : (bool)
        specify extra info. Default is 0 
    '''
    sums = [] 
    for stat in sumstat: 
        if stat == 'smf': # central stellar mass function 
            try:
                m_arr, smf = Obvs.getMF(cencat['m.star'], weights=cencat['weights'])
            except ValueError: 
                smf = np.zeros(38)
            sums.append(smf) 
        else: 
            raise NotImplementedError
    return sums


def runABC(run, T, eps0, prior, N_p=1000, sumstat=None, restart=False, t_restart=None, nsnap0=15, downsampled='20'): 
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

    prior : (object)
        object that specifies the priors
    '''
    if len(eps0) != len(sumstat): 
        raise ValueError('Epsilon thresholds should correspond to number of summary statistics')
    
    # summary statistics of data 
    data_sum = dataSum(sumstat=sumstat)

    # get uncertainties of central SMF
    m_arr, _, phi_err = Obvs.dataSMF(source='li-white')
    # now scale err by f_cen 
    phi_err *= np.sqrt(1./(1.-np.array([Obvs.f_sat(mm, 0.05) for mm in m_arr])))

    # summary statistics of simulation 
    def Sim(tt): 
        cencat = model(run, tt, nsnap0=nsnap0, downsampled=downsampled)
        sums = modelSum(cencat, sumstat=sumstat)
        return sums 

    # distance metric 
    def Rho(simsum, datsum): 
        nonzero = np.where((simsum[0] > 0.) & (datsum[0] > 0.)) # preventing nans  
        n_bins = float(len(nonzero[0])) # number of bins 
        return np.sum((simsum[0][nonzero] - datsum[0][nonzero])**2/(phi_err[nonzero]**2))/n_bins

    init_pool = None 
    # for restarting ABC (use with caution)
    if restart: 
        if t_restart is None: 
            raise ValueError("specify restart iteration number") 
        abcout = readABC(run, t_restart)
        init_pool = abcpmc.PoolSpec(t_restart, None, None, abcout['theta'], abcout['rho'], abcout['w'])

    # threshold 
    eps = abcpmc.ConstEps(T, eps0)
    try:
        mpi_pool = mpi_util.MpiPool()
        abcpmc_sampler = abcpmc.Sampler(
                N=N_p,          # N_particles
                Y=data_sum,     # data
                postfn=Sim,     # simulator 
                dist=Rho,       # distance function  
                pool=mpi_pool)  
        print('yes MPI') 
    except AttributeError: 
        print('no MPI') 
        abcpmc_sampler = abcpmc.Sampler(
                N=N_p,          # N_particles
                Y=data_sum,     # data
                postfn=Sim,     # simulator 
                dist=Rho)       # distance function  

    # Write out all details of the run info 
    write_kwargs = {} 
    write_kwargs['Niter'] = T
    write_kwargs['sumstat'] = sumstat 
    write_kwargs['prior'] = prior
    write_kwargs['nsnap0'] = nsnap0 # initial snapshot
    write_kwargs['downsampled'] = downsampled # downsample factor
    if not restart: 
        Writeout('init', run, abcpmc_sampler, **write_kwargs)
    else: 
        Writeout('restart', run, abcpmc_sampler, **write_kwargs)

    # particle proposal 
    abcpmc_sampler.particle_proposal_cls = abcpmc.ParticleProposal

    pools = []
    if init_pool is None:   # initiate epsilon write out
        Writeout('eps', run, None)

    print('----------------------------------------')
    for pool in abcpmc_sampler.sample(prior, eps):#, pool=init_pool):
        print("T:{0},ratio: {1:>.4f}".format(pool.t, pool.ratio))
        print('eps ', eps(pool.t))
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
        print('----------------------------------------')
        pools.append(pool)

    return pools 


def minimize(run, sumstat=None, **run_kwargs): 
    '''
    '''
    # summary statistics of data 
    data_sum = dataSum(sumstat=sumstat)

    # get uncertainties of central SMF
    m_arr, _, phi_err = Obvs.dataSMF(source='li-white')
    # now scale err by f_cen 
    phi_err *= np.sqrt(1./(1.-np.array([Obvs.f_sat(mm, 0.05) for mm in m_arr])))

    # summary statistics of simulation 
    sim_kwargs = {} 
    sim_kwargs['nsnap0'] = run_kwargs['nsnap0']
    sim_kwargs['downsampled'] = run_kwargs['downsampled']
    def chi2(tt): 
        cencat = model(run, tt, **sim_kwargs)
        simsum = modelSum(cencat, sumstat=sumstat)
        nonzero = np.where((simsum[0] > 0.) & (data_sum[0] > 0.)) # preventing nans  
        n_bins = float(len(nonzero[0])) # number of bins 
        return np.sum((simsum[0][nonzero] - data_sum[0][nonzero])**2/(phi_err[nonzero]**2))/n_bins
    
    for m_m in [0.5, 1., 2., 2.5]: 
        for m_z in [-0.5, -0.25, 0., 0.25, 0.5]: 
            print('[%f, %f] -- chi2 = %f' % (m_m, m_z, chi2((m_m, m_z))))
    theta_opt = sp.optimize.minimize(chi2, [1.5, 0.1], method='L-BFGS-B', bounds=[[0.5, 2.5], [-0.5, 0.5]]) 
    print theta_opt
    return None 


def Writeout(type, run, pool, **kwargs): 
    ''' Given abcpmc pool object. Writeout specified ABC pool property
    '''
    abc_dir = ''.join([UT.dat_dir(), 'abc/', run, '/']) 

    if type == 'init': # initialize
        if not os.path.exists(abc_dir): # make directory if it doesn't exist 
            try: 
                os.makedirs(abc_dir)
            except OSError: 
                pass 
        
        # write specific info of the run  
        f = open(abc_dir+'info.md', 'w')
        f.write('# '+run+' run specs \n')
        f.write(''.join(['N_iter = ', str(kwargs['Niter']), '\n']))
        f.write(''.join(['N_particles = ', str(pool.N), '\n']))
        f.write(''.join(['Distance function = ', pool.dist.__name__ , '\n']))
        # prior 
        prior_obj = kwargs['prior'] 
        f.write('Top Hat Priors \n')
        f.write(''.join(['Prior Min = [', ','.join([str(prior_obj.min[i]) for i in range(len(prior_obj.min))]), '] \n']))
        f.write(''.join(['Prior Max = [', ','.join([str(prior_obj.max[i]) for i in range(len(prior_obj.max))]), '] \n']))
        f.write('\n') 

        f.write(''.join(['Initial Snapshot = ', str(kwargs['nsnap0']), '\n']))
        f.write(''.join(['Downsampled by = ', str(kwargs['downsampled']), '\n']))
        f.close()
    elif type == 'restart': # initialize
        if not os.path.exists(abc_dir+'info.md'): # make directory if it doesn't exist 
            raise ValueError('cannot find run directory')
        
        # write specific info of the run  
        f = open(abc_dir+'info.md', 'a')
        f.write('# RESTARTING, details below should agree with details above')
        f.write('# '+run+' run specs \n')
        f.write(''.join(['N_iter = ', str(kwargs['Niter']), '\n']))
        f.write(''.join(['N_particles = ', str(pool.N), '\n']))
        f.write(''.join(['Distance function = ', pool.dist.__name__ , '\n']))
        # prior 
        prior_obj = kwargs['prior'] 
        f.write('Top Hat Priors \n')
        f.write(''.join(['Prior Min = [', ','.join([str(prior_obj.min[i]) for i in range(len(prior_obj.min))]), '] \n']))
        f.write(''.join(['Prior Max = [', ','.join([str(prior_obj.max[i]) for i in range(len(prior_obj.max))]), '] \n']))
        f.write('\n') 

        f.write(''.join(['Initial Snapshot = ', str(kwargs['nsnap0']), '\n']))
        f.write(''.join(['Downsampled by = ', str(kwargs['downsampled']), '\n']))
        f.close()
    elif type == 'eps': # threshold writeout 
        if pool is None: # write or overwrite threshold writeout
            f = open(''.join([abc_dir, 'epsilon.', run, '.dat']), "w")
        else: 
            f = open(''.join([abc_dir, 'epsilon.', run, '.dat']), "a") #append 
            f.write(str(pool.eps)+'\t'+str(pool.ratio)+'\n')
        f.close()
    elif type == 'theta': # particle thetas
        np.savetxt(''.join([abc_dir, 'theta.t', str(pool.t), '.', run, '.dat']), pool.thetas) 
    elif type == 'w': # particle weights
        np.savetxt(''.join([abc_dir, 'w.t', str(pool.t), '.', run, '.dat']), pool.ws)
    elif type == 'rho': # distance
        np.savetxt(''.join([abc_dir, 'rho.t', str(pool.t), '.', run, '.dat']), pool.dists)
    else: 
        raise ValueError
    return None 


def readABC(run, T): 
    ''' Read in theta, w, and rho from ABC writeouts
    '''
    file = lambda ss, t, r: ''.join([UT.dat_dir(), 'abc/', run, '/', ss, '.t', str(t), '.', r, '.dat'])
    abc_out = {}    # read in theta, w, rho 
    abc_out['theta'] = np.loadtxt(file('theta', T, run)) 
    abc_out['w'] = np.loadtxt(file('w', T, run))
    abc_out['rho'] = np.loadtxt(file('rho', T, run))
    return abc_out


def model_ABCparticle(run, T, nsnap0=15, sigma_smhm=0.2): 
    ''' Evaluate and save specific columns of the forward model evaluated for each of the 
    particles in the T-th iteration of the ABC run. Takes... a while 
    '''
    # read in the abc particles 
    abcout = readABC(run, T)
    abc_dir = UT.dat_dir()+'abc/'+run+'/model/' # directory where all the ABC files are stored
    
    # save the median theta separately (evaluate it a bunch of times) 
    #theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) for i in range(len(abcout['theta'][0]))]
    theta_med = [np.median(abcout['theta'][:,i]) for i in range(abcout['theta'].shape[1])]
    for i in range(10):  
        subcat_sim = model(run, theta_med, nsnap0=nsnap0, sigma_smhm=sigma_smhm, downsampled='14') 

        fname = ''.join([abc_dir, 'model.theta_median', str(i), '.t', str(T), '.hdf5'])
        f = h5py.File(fname, 'w') 
        for key in ['m.star', 'halo.m', 'm.max', 'weights', 'sfr', 'gclass']: 
            f.create_dataset(key, data=subcat_sim[key])
        f.close()
    #return None  
    # now save the rest 
    for i in range(len(abcout['w'])): 
        subcat_sim_i = model(run, abcout['theta'][i], nsnap0=nsnap0, sigma_smhm=sigma_smhm, downsampled='14') 
        fname = ''.join([abc_dir, 'model.theta', str(i), '.t', str(T), '.hdf5'])
        f = h5py.File(fname, 'w') 
        for key in ['m.star', 'halo.m', 'm.max', 'weights', 'sfr', 'gclass']: 
            f.create_dataset(key, data=subcat_sim_i[key])
        f.close()
    return None  
