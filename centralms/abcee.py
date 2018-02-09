'''

use ABC-PMC 

'''
import os 
import h5py 
import numpy as np
import abcpmc
from abcpmc import mpi_util
import corner as DFM
try: 
    import codif
    flag_codif = True 
except ImportError: 
    flag_codif = False

# -- local -- 
import observables as Obvs
import util as UT
import catalog as Cat
import evolver as Evol
import sfh as SFH

import matplotlib.pyplot as plt 


def Theta(run): 
    tt = {} 
    #tt['variable'] = ['SFMS z slope', 'SFMS m slope']#, 'SFMS offset']
    #tt['label'] = ['$m_{z; SFMS}$', '$m_{M_*; SFMS}$']#, '$c_\mathrm{SFMS}$']
    tt['variable'] = ['SFMS amp z param', 'SFMS slope z param']
    tt['label'] = ['$m_{z; amp}$', '$m_{z; slope}$']
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

    # SFMS_zslope, SFMS_mslope
    #prior_min = [1., 0.4]#, -0.15]
    #prior_max = [1.8, 0.8]#, -0.06]

    # new priors since we implemented "anchored" SFMS  
    # SFMS amplitude z-dep parameter, SFMS slope z-dep parameter
    prior_min = [0.5, -0.5]#, -0.15]
    prior_max = [2., 0.5]#, -0.06]

    prior_obj = abcpmc.TophatPrior(prior_min, prior_max) 

    return prior_obj


def Data(**data_kwargs): 
    ''' Our 'data'
    '''
    subhist = Cat.PureCentralSubhalos(nsnap0=data_kwargs['nsnap0'], 
            sigma_smhm=data_kwargs['sigma_smhm'])
    subcat = subhist.Read(downsampled='14') # full/downsampled does not make a difference 
    return subcat


def SumData(sumstat, m_arr=np.linspace(9.5, 12.0, 11), info=False, **data_kwargs): 
    ''' Return the summary statistics of data 
    '''
    subcat = Data(**data_kwargs)

    sums = [] 
    if 'smf' in sumstat: # stellar mass function 
        if 'nsnap0' not in data_kwargs.keys():
            raise ValueError

        smf = Obvs.getMF(subcat['m.star'], weights=subcat['weights'], m_arr=m_arr) 

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
    # args = SFMS_zslope, SFMS_mslope

    # these values were set by cenque project's output
    theta['gv'] = {'slope': 1.03, 'fidmass': 10.5, 'offset': -0.02}
    theta['fq'] = {'name': 'cosmos_tinker'}
    theta['fpq'] = {'slope': -2.079703, 'offset': 1.6153725, 'fidmass': 10.5}
    
    theta['mass'] = {'solver': 'euler', 'f_retain': 0.6, 't_step': 0.05} 

    # SFMS slopes can change 
    #theta['sfms'] = {'zslope': args[0], 'mslope': args[1]}#, 'offset': args[2]}
    theta['sfms'] = {'name': 'anchored', 'amp': args[0], 'slope': args[1], 'sigma': 0.3}

    if run == 'test0': 
        # simplest test with constant offset SFH 
        theta['sfh'] = {'name': 'constant_offset'}
        theta['sfh']['nsnap0'] =  kwargs['nsnap0'] 
    elif run == 'randomSFH':  
        # random fluctuation SFH where fluctuations 
        # happen on fixed 0.5 Gyr timescales  
        theta['sfh'] = {'name': 'random_step_fluct'} 
        theta['sfh']['dt_min'] = 0.5 
        theta['sfh']['dt_max'] = 0.5 
        theta['sfh']['sigma'] = 0.3 
    elif run == 'randomSFH_1gyr':  
        # random fluctuation SFH where fluctuations 
        # happen on fixed 1 Gyr timescales  
        theta['sfh'] = {'name': 'random_step_fluct'} 
        theta['sfh']['dt_min'] = 1. 
        theta['sfh']['dt_max'] = 1. 
        theta['sfh']['sigma'] = 0.3 
    elif run == 'randomSFH_2gyr':  
        # random fluctuation SFH where fluctuations 
        # happen on fixed 2 Gyr timescales  
        theta['sfh'] = {'name': 'random_step_fluct'} 
        theta['sfh']['dt_min'] = 2. 
        theta['sfh']['dt_max'] = 2. 
        theta['sfh']['sigma'] = 0.3 
    elif run == 'randomSFH_5gyr':  
        # random fluctuation SFH where fluctuations 
        # happen on fixed 2 Gyr timescales  
        theta['sfh'] = {'name': 'random_step_fluct'} 
        theta['sfh']['dt_min'] = 5. 
        theta['sfh']['dt_max'] = 5. 
        theta['sfh']['sigma'] = 0.3 
    elif run == 'randomSFH_10gyr':  
        # random fluctuation SFH where fluctuations 
        # happen on fixed 2 Gyr timescales  
        theta['sfh'] = {'name': 'random_step_fluct'} 
        theta['sfh']['dt_min'] = 10. 
        theta['sfh']['dt_max'] = 10. 
        theta['sfh']['sigma'] = 0.3 
    elif run == 'randomSFH_0.5gyr':  
        # random fluctuation SFH where fluctuations 
        # happen on fixed 2 Gyr timescales  
        theta['sfh'] = {'name': 'random_step_fluct'} 
        theta['sfh']['dt_min'] = 0.5
        theta['sfh']['dt_max'] = 0.5 
        theta['sfh']['sigma'] = 0.3 
        theta['mass']['t_step'] = 0.025 # change timestep 
    elif run == 'randomSFH_integtest':  
        # random fluctuation SFH where fluctuations 
        # happen on fixed 0.5 Gyr timescales  
        theta['sfh'] = {'name': 'random_step_fluct'} 
        theta['sfh']['dt_min'] = 0.5 
        theta['sfh']['dt_max'] = 0.5 
        theta['sfh']['sigma'] = 0.3 
        theta['mass']['t_step'] = 0.01 # change timestep 
    elif run == 'randomSFH_short':  
        # random fluctuation SFH where fluctuations 
        # happen on fixed short 0.1 Gyr timescales  
        theta['sfh'] = {'name': 'random_step_fluct'} 
        theta['sfh']['dt_min'] = 0.1 
        theta['sfh']['dt_max'] = 0.1 
        theta['sfh']['sigma'] = 0.3 
        theta['mass']['t_step'] = 0.01 # change timestep 
    elif run == 'randomSFH_long':  
        # random fluctuation SFH where fluctuations 
        # happen on fixed longer 1 Gyr timescales  
        theta['sfh'] = {'name': 'random_step_fluct'} 
        theta['sfh']['dt_min'] = 5.
        theta['sfh']['dt_max'] = 5. 
        theta['sfh']['sigma'] = 0.3 
    elif run == 'rSFH_r1.0_most': 
        theta['sfh'] = {'name': 'random_step_most_abias'}
        theta['sfh']['dt_min'] = 0.5 
        theta['sfh']['dt_max'] = 0.5 
        theta['sfh']['sigma_tot'] = 0.3 
        theta['sfh']['sigma_corr'] = 0.3
    elif run == 'randomSFH_r0.2': 
        # random fluctuation SFH corrected by r=0.2 with halo aseembly property 
        # fluctuations happen on fixed 0.5 Gyr timescales  
        # halo assembly property here is halo mass growth over 2 Gyrs 
        theta['sfh'] = {'name': 'random_step_abias2'} 
        theta['sfh']['dt_min'] = 0.5 
        theta['sfh']['dt_max'] = 0.5 
        theta['sfh']['t_abias'] = 2. # Gyr
        theta['sfh']['sigma_tot'] = 0.3 
        theta['sfh']['sigma_corr'] = 0.2 * 0.3
    elif run == 'randomSFH_r0.99': 
        # random fluctuation SFH corrected by r=0.99 with halo aseembly property 
        # fluctuations happen on fixed 0.5 Gyr timescales  
        # halo assembly property here is halo mass growth over 2 Gyrs 
        theta['sfh'] = {'name': 'random_step_abias2'} 
        theta['sfh']['dt_min'] = 0.5 
        theta['sfh']['dt_max'] = 0.5 
        theta['sfh']['t_abias'] = 2. # Gyr
        theta['sfh']['sigma_tot'] = 0.3 
        theta['sfh']['sigma_corr'] = 0.99 * 0.3
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
    elif run == 'rSFH_r0.99_tdyn_0.5gyr': 
        # random SFH with 0.99 correlation with halo growth 
        # over t_dyn and duty cycle of 0.5 Gyr 
        theta['sfh'] = {'name': 'random_step_abias_delay_dt'}
        theta['sfh']['dt_min'] = 0.5 
        theta['sfh']['dt_max'] = 0.5 
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
    elif run == 'randomSFH_5gyr_narrSFMS':  
        # random fluctuation SFH where fluctuations happen on fixed 5 Gyr timescales  
        # we also use a 0.2 dex scatter SFMS. 
        theta['sfh'] = {'name': 'random_step_fluct'} 
        theta['sfh']['dt_min'] = 5. 
        theta['sfh']['dt_max'] = 5. 
        theta['sfh']['sigma'] = 0.2  # note narrower SFMS
        theta['sfms']['sigma'] = 0.2 # note narrower SFMS
    else: 
        raise NotImplementedError

    # load in Subhalo Catalog (pure centrals)
    if 'sigma_smhm' in kwargs.keys(): 
        subhist = Cat.PureCentralSubhalos(nsnap0=kwargs['nsnap0'], 
                sigma_smhm=kwargs['sigma_smhm'])
    else: 
        subhist = Cat.PureCentralSubhalos(nsnap0=kwargs['nsnap0'])
    subcat = subhist.Read(downsampled=kwargs['downsampled']) # halo sample  
    
    eev = Evol.Evolver(subcat, theta, nsnap0=kwargs['nsnap0'])
    eev.InitSF()
    if 'forTests' not in kwargs.keys(): 
        eev.newEvolve() 
        return eev.SH_catalog
    else: 
        if kwargs['forTests']: 
            eev.newEvolve(forTests=True) 
            return eev.SH_catalog, eev


def SumSim(sumstat, subcat, info=False, m_arr=np.linspace(9.5, 12.0, 11)): #, **sim_kwargs): 
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
            try:
                smf = Obvs.getMF(subcat['m.star'], weights=subcat['weights'], m_arr=m_arr)
            except ValueError: 
                smf = [m_arr, np.zeros(len(m_arr))]

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


# different distance metric calculations 
def roe_wrap(sumstat, type='L2'):
    ''' Get it? Wrapper for Rhos or roe wrap. 
    '''
    if len(sumstat) == 1: # only SMF 
        if type == 'L2': 
            return L2_logSMF
    else: 
        raise NotImplementedError


def L2_logSMF(simsum, datsum): 
    ''' Measure the L2 norm for the case where the summary statistic 
    is the log(SMF). 
    '''
    if len(simsum[0]) != len(datsum[0]): 
        raise ValueError

    nonzero = np.where((simsum[0] > 0.) & (datsum[0] > 0.)) # preventing nans  
    n_bins = len(nonzero[0]) # number of bins 

    return np.sum((np.log10(simsum[0][nonzero]) - np.log10(datsum[0][nonzero]))**2)/np.float(n_bins)


def runABC(run, T, eps0, N_p=1000, sumstat=None, notify=False, 
        restart=False, t_restart=None, **run_kwargs): 
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
    m_arr = np.linspace(9.5, 12.0, 11) 
    data_kwargs = {} 
    data_kwargs['nsnap0'] = run_kwargs['nsnap0']
    data_kwargs['sigma_smhm'] = 0.2 #run_kwargs['sigma_smhm']
    data_sum = SumData(sumstat, m_arr=m_arr, **data_kwargs)

    # summary statistics of simulation 
    sim_kwargs = {} 
    sim_kwargs['nsnap0'] = run_kwargs['nsnap0']
    sim_kwargs['downsampled'] = run_kwargs['downsampled']
    def Sim(tt): 
        sh_catalog = model(run, tt, **sim_kwargs)
        sums = SumSim(sumstat, sh_catalog, m_arr=m_arr)
        return sums 

    # get uncertainties of central SMF
    _, _, phi_err = Obvs.MF_data(source='li-white', m_arr=m_arr)
    # now scale err by f_cen 
    phi_err *= np.sqrt(1./(1.-np.array([Obvs.f_sat(mm, 0.05) for mm in m_arr])))

    # distance metric 
    def Rho(simsum, datsum): 
        nonzero = np.where((simsum[0] > 0.) & (datsum[0] > 0.)) # preventing nans  
        n_bins = len(nonzero[0]) # number of bins 
	#print np.sum((simsum[0][nonzero] - datsum[0][nonzero])**2/(phi_err[nonzero]**2))/float(n_bins)
        return np.sum((simsum[0][nonzero] - datsum[0][nonzero])**2/(phi_err[nonzero]**2))/float(n_bins)

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
                N=N_p,                # N_particles
                Y=data_sum,             # data
                postfn=Sim,             # simulator 
                dist=Rho,               # distance function  
                pool=mpi_pool)  

    except AttributeError: 
        abcpmc_sampler = abcpmc.Sampler(
                N=N_p,                # N_particles
                Y=data_sum,             # data
                postfn=Sim,            # simulator 
                dist=Rho)           # distance function  

    # Write out all details of the run info 
    write_kwargs = {} 
    write_kwargs['Niter'] = T
    write_kwargs['sumstat'] = sumstat 
    for key in run_kwargs.keys():
        write_kwargs[key] = run_kwargs[key]
    if not restart: 
        Writeout('init', run, abcpmc_sampler, **write_kwargs)
    else: 
        Writeout('restart', run, abcpmc_sampler, **write_kwargs)

    # particle proposal 
    abcpmc_sampler.particle_proposal_cls = abcpmc.ParticleProposal

    pools = []
    if init_pool is None:   # initiate epsilon write out
        Writeout('eps', run, None)

    print '----------------------------------------'
    for pool in abcpmc_sampler.sample(prior_obj, eps):#, pool=init_pool):
        print("T:{0},ratio: {1:>.4f}".format(pool.t, pool.ratio))
        print 'eps ', eps(pool.t)
        Writeout('eps', run, pool)

        # write out theta, weights, and distances to file 
        Writeout('theta', run, pool) 
        Writeout('w', run, pool) 
        Writeout('rho', run, pool) 
        #plotABC(run, pool.t) # plot corner plot 

        if notify and flag_codif: 
            codif.notif(subject=run+' T = '+str(pool.t)+' FINISHED')
        # update epsilon based on median thresholding 
        if len(eps0) > 1: 
            eps.eps = np.median(np.atleast_2d(pool.dists), axis = 0)
        else: 
            eps.eps = np.median(pool.dists)
        print '----------------------------------------'
        pools.append(pool)

    if notify and flag_codif: 
        codif.notif(subject=run+' ALL FINISHED')
    return pools 


def Writeout(type, run, pool, **kwargs): 
    ''' Given abcpmc pool object. Writeout specified ABC pool property
    '''
    file = UT.dat_dir()+'abc/'+run+'/'

    if type == 'init': # initialize
        if not os.path.exists(file): # make directory if it doesn't exist 
            try: 
                os.makedirs(file)
            except OSError: 
                pass 
        
        # write specific info of the run  
        file += 'info.md'
        f = open(file, 'w')
        f.write('# '+run+' run specs \n')
        f.write(''.join(['N_iter = ', str(kwargs['Niter']), '\n']))
        f.write(''.join(['N_particles = ', str(pool.N), '\n']))
        f.write(''.join(['Distance function = ', pool.dist.__name__ , '\n']))
        # variables
        theta_info = Theta(run)
        f.write(''.join(['Variables = [', ','.join(theta_info['variable']), '] \n']))
        # prior 
        prior_obj = Prior(run)
        f.write('Top Hat Priors \n')
        f.write(''.join(['Prior Min = [', ','.join([str(prior_obj.min[i]) for i in range(len(prior_obj.min))]), '] \n']))
        f.write(''.join(['Prior Max = [', ','.join([str(prior_obj.max[i]) for i in range(len(prior_obj.max))]), '] \n']))
        f.write('\n') 

        f.write(''.join(['Initial Snapshot = ', str(kwargs['nsnap0']), '\n']))
        f.write(''.join(['Downsampled by = ', str(kwargs['downsampled']), '\n']))
        f.close()
    elif type == 'restart': # initialize
        if not os.path.exists(file): # make directory if it doesn't exist 
            raise ValueError('cannot find run directory')
        
        # write specific info of the run  
        file += 'info.md'
        f = open(file, 'a')
        f.write('# RESTARTING, details below should agree with details above')
        f.write('# '+run+' run specs \n')
        f.write(''.join(['N_iter = ', str(kwargs['Niter']), '\n']))
        f.write(''.join(['N_particles = ', str(pool.N), '\n']))
        f.write(''.join(['Distance function = ', pool.dist.__name__ , '\n']))
        # variables
        theta_info = Theta(run)
        f.write(''.join(['Variables = [', ','.join(theta_info['variable']), '] \n']))
        # prior 
        prior_obj = Prior(run)
        f.write('Top Hat Priors \n')
        f.write(''.join(['Prior Min = [', ','.join([str(prior_obj.min[i]) for i in range(len(prior_obj.min))]), '] \n']))
        f.write(''.join(['Prior Max = [', ','.join([str(prior_obj.max[i]) for i in range(len(prior_obj.max))]), '] \n']))
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
    return None 


def readABC(run, T): 
    ''' Read in theta, w, and rho from ABC writeouts
    '''
    dir = UT.dat_dir()+'abc/'+run+'/'
    file = lambda ss, t, r: ''.join([dir, ss, '.t', str(t), '.', r, '.dat'])
    
    abc_out = {}    # read in theta, w, rho 
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


def qaplotABC(run, T, sumstat=['smf'], nsnap0=15, sigma_smhm=0.2, downsampled='14', theta=None, figure=None): 
    ''' Quality assurance plot for ABC runs. Plot summary statistic(s), SMHMR, SFMS
    '''
    # first compare data summary statistics to Sim(median theta) 
    if theta is None:
        abcout = readABC(run, T)
        # median theta 
        theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) for i in range(len(abcout['theta'][0]))]
    else: 
        theta_med = theta

    subcat_dat = Data(nsnap0=nsnap0, sigma_smhm=sigma_smhm) # 'data'
    sumdata = SumData(sumstat, info=True, nsnap0=nsnap0, sigma_smhm=sigma_smhm)  

    subcat_sim = model(run, theta_med, 
            nsnap0=nsnap0, sigma_smhm=sigma_smhm, downsampled=downsampled) 
    sumsim = SumSim(sumstat, subcat_sim, info=True)
    
    fig = plt.figure(figsize=(6*(len(sumstat)+3),5))
    
    for i_s, stat in enumerate(sumstat): 
        if stat == 'smf': 
            sub = fig.add_subplot(1, len(sumstat)+3, i_s+1)

            sub.plot(sumdata[0][0], sumdata[0][1], c='k', ls='--', label='Data')
            sub.plot(sumsim[0][0], sumsim[0][1], c='b', label='Sim.')

            sub.set_xlim([9., 12.])
            sub.set_xlabel('$log\;M_*$', fontsize=25)
            sub.set_ylim([1e-6, 10**-1.75])
            sub.set_yscale('log')
            sub.set_ylabel('$\Phi$', fontsize=25)
            sub.legend(loc='upper right') 
        else: 
            raise NotImplementedError
    
    # SMHMR panel of SF galaxies  
    isSF = np.where(subcat_sim['gclass'] == 'sf') # only SF galaxies 

    sub = fig.add_subplot(1, len(sumstat)+3, len(sumstat)+1)
    smhmr = Obvs.Smhmr()
    # simulation 
    m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_sim['halo.m'][isSF], subcat_sim['m.star'][isSF], 
            dmhalo=0.2, weights=subcat_sim['weights'][isSF])
    sub.plot(m_mid, sig_mhalo, c='#1F77B4', lw=2, label='Model') 
    sig_sim = sig_mhalo[np.argmin(np.abs(m_mid-12.))]
    # data 
    m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_sim['halo.m'][isSF], subcat_sim['m.sham'][isSF], 
            dmhalo=0.2, weights=subcat_dat['weights'][isSF])
    enough = (cnts > 50) 
    sub.plot(m_mid[enough], sig_mhalo[enough], c='k', ls='--', label='SHAM') 
    sig_dat = sig_mhalo[np.argmin(np.abs(m_mid-12.))]
    
    #sig_dat = smhmr.sigma_logMstar(subcat_sim['halo.m'][isSF], subcat_sim['m.sham'][isSF], 
    #        weights=subcat_sim['weights'][isSF], dmhalo=0.2)
    #sig_sim = smhmr.sigma_logMstar(subcat_sim['halo.m'][isSF], subcat_sim['m.star'][isSF], 
    #        weights=subcat_sim['weights'][isSF], dmhalo=0.2)

    # mark sigma_M*(M_h = 10^12) 
    sub.text(0.95, 0.9, 
            ''.join(['$\sigma^{(s)}_{M_*}(M_h = 10^{12} M_\odot) = ', str(round(sig_sim,2)), '$ \n', 
                '$\sigma^{(d)}_{M_*}(M_h = 10^{12} M_\odot) = ', str(round(sig_dat,2)), '$']), 
            fontsize=15, ha='right', va='top', transform=sub.transAxes)

    sub.set_xlim([10., 15.])
    sub.set_xlabel('$log\;M_{halo}$', fontsize=25)
    sub.set_ylim([0., 0.6])
    sub.set_ylabel('$\sigma_{log\,M_*}$', fontsize=25)

    # SFMS panel 
    sub = fig.add_subplot(1, len(sumstat)+3, len(sumstat)+2)
    DFM.hist2d(
            subcat_sim['m.star'][isSF], 
            subcat_sim['sfr'][isSF], 
            weights=subcat_sim['weights'][isSF], 
            levels=[0.68, 0.95], range=[[8., 12.], [-4., 2.]], color='#1F77B4', 
            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 

    # observations 
    m_arr = np.arange(8., 12.1, 0.1)
    sfr_arr = SFH.SFR_sfms(m_arr, UT.z_nsnap(1), subcat_sim['theta_sfms'])
    sub.plot(m_arr, sfr_arr+0.3, ls='--', c='k') 
    sub.plot(m_arr, sfr_arr-0.3, ls='--', c='k') 

    sub.set_xlim([8., 12.])
    sub.set_xlabel('$\mathtt{log\;M_*}$', fontsize=25)
    sub.set_ylim([-4., 2.])
    sub.set_ylabel('$\mathtt{log\;SFR}$', fontsize=25)
    
    # dSFR as a function of t_cosmic 
    sub = fig.add_subplot(1, len(sumstat)+3, len(sumstat)+3)
    mbins = np.arange(9., 12., 0.5) 

    i_r = [] # select random SF galaxies over mass bins
    for i_m in range(len(mbins)-1): 
        inmbin = np.where(
                (subcat_sim['gclass'] == 'sf') & 
                (subcat_sim['m.star'] > mbins[i_m]) & 
                (subcat_sim['m.star'] <= mbins[i_m+1]))

        i_r.append(np.random.choice(inmbin[0], size=1)[0])
    i_r = np.array(i_r)

    # calculate d(logSFR)  = logSFR - logSFR_MS 
    dlogsfrs = np.zeros((len(i_r), nsnap0-1))
    for i_snap in range(1, nsnap0): 
        if i_snap == 1: 
            sfr = subcat_sim['sfr'][i_r]
            sfr_ms = SFH.SFR_sfms(subcat_sim['m.star'][i_r], UT.z_nsnap(i_snap), subcat_sim['theta_sfms']) 
            dlogsfrs[:,0] =  sfr - sfr_ms 
        else: 
            sfr = subcat_sim['sfr.snap'+str(i_snap)][i_r]
            sfr_ms = SFH.SFR_sfms(subcat_sim['m.star.snap'+str(i_snap)][i_r], UT.z_nsnap(i_snap), subcat_sim['theta_sfms']) 
            dlogsfrs[:,i_snap-1] = sfr - sfr_ms 

    for i in range(dlogsfrs.shape[0]): 
        sub.plot(UT.t_nsnap(range(1, nsnap0)), dlogsfrs[i,:]) 
    for i in range(1, nsnap0): 
        sub.vlines(UT.t_nsnap(i), -1., 1., color='k', linestyle='--')

    sub.set_xlim([13.81, 9.])
    sub.set_xticks([13., 12., 11., 10., 9.])
    sub.set_xlabel('$\mathtt{t_{cosmic}\;[Gyr]}$', fontsize=25)
    sub.set_ylim([-1., 1.]) 
    sub.set_yticks([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9])
    sub.set_ylabel('$\mathtt{\Delta log\,SFR}$', fontsize=25)

    if theta is None: 
        fig_name = ''.join([UT.dat_dir()+'abc/'+run+'/', 'qaplot.t', str(T), '.', run, '.png'])
        fig.savefig(fig_name, bbox_inches='tight')
        plt.close()
    else: 
        if figure is None: 
            plt.show() 
            plt.close() 
        else: 
            fig.savefig(figure, bbox_inches='tight')
            plt.close()
    return None 


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
