'''

use ABC-PMC 

'''
import os 
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

import matplotlib.pyplot as plt 

def Theta(run): 
    tt = {} 
    if run in ['test0', 'randomSFH', 'randomSFH_short', 'randomSFH_long', 'randomSFH_r0.2', 'randomSFH_r0.99', 
            'rSFH_r0.66_delay', 'rSFH_r0.99_delay', 'rSFH_r1.0_most']: 
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

    if run in ['test0', 'randomSFH', 'randomSFH_short', 'randomSFH_long', 'randomSFH_r0.2', 'randomSFH_r0.99', 
            'rSFH_r0.66_delay', 'rSFH_r0.99_delay', 'rSFH_r1.0_most']: 
        # SFMS_zslope, SFMS_mslope
        prior_min = [0.9, 0.4]
        prior_max = [1.5, 0.7]
    else:
        raise NotImplementedError

    prior_obj = abcpmc.TophatPrior(prior_min, prior_max) 

    return prior_obj


def Data(**data_kwargs): 
    ''' Our 'data'
    '''
    subhist = Cat.PureCentralHistory(nsnap_ancestor=data_kwargs['nsnap0'], 
            sigma_smhm=data_kwargs['sigma_smhm'])
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

    if run in ['test0', 'randomSFH', 'randomSFH_short', 'randomSFH_long', 'randomSFH_r0.2', 'randomSFH_r0.99', 
            'rSFH_r0.66_delay', 'rSFH_r0.99_delay', 'rSFH_r1.0_most']: 
        # args = SFMS_zslope, SFMS_mslope

        # these values were set by cenque project's output
        theta['gv'] = {'slope': 1.03, 'fidmass': 10.5, 'offset': -0.02}
        theta['fq'] = {'name': 'cosmos_tinker'}
        theta['fpq'] = {'slope': -2.079703, 'offset': 1.6153725, 'fidmass': 10.5}
        
        theta['mass'] = {'solver': 'euler', 'f_retain': 0.6, 't_step': 0.05} 

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
            theta['sfh']['dt_min'] = 1.
            theta['sfh']['dt_max'] = 1. 
            theta['sfh']['sigma'] = 0.3 
        elif run == 'rSFH_r1.0_most': 
            theta['sfh'] = {'name': 'random_step_most_abias'}
            theta['sfh']['dt_min'] = 5. 
            theta['sfh']['dt_max'] = 5. 
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
            theta['sfh'] = {'name': 'random_step_abias_delay'}
            theta['sfh']['dt_min'] = 0.5 
            theta['sfh']['dt_max'] = 0.5 
            theta['sfh']['sigma_tot'] = 0.3 
            theta['sfh']['sigma_corr'] = 0.66 * 0.3
            theta['sfh']['dt_delay'] = 1. # Gyr 
            theta['sfh']['dz_dMh'] = 0.5 
        elif run == 'rSFH_r0.99_delay': 
            theta['sfh'] = {'name': 'random_step_abias_delay'}
            theta['sfh']['dt_min'] = 0.5 
            theta['sfh']['dt_max'] = 0.5 
            theta['sfh']['sigma_tot'] = 0.3 
            theta['sfh']['sigma_corr'] = 0.99 * 0.3
            theta['sfh']['dt_delay'] = 1. # Gyr 
            theta['sfh']['dz_dMh'] = 0.5 

        # SFMS slopes can change 
        theta['sfms'] = {'name': 'linear', 'zslope': args[0], 'mslope': args[1]}

        # load in Subhalo Catalog (pure centrals)
        if 'sigma_smhm' in kwargs.keys(): 
            subhist = Cat.PureCentralHistory(nsnap_ancestor=kwargs['nsnap0'], 
                    sigma_smhm=kwargs['sigma_smhm'])
        else: 
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
    data_kwargs = {} 
    data_kwargs['nsnap0'] = run_kwargs['nsnap0']
    data_kwargs['sigma_smhm'] = run_kwargs['sigma_smhm']

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
    for pool in abcpmc_sampler.sample(prior_obj, eps, pool=init_pool):
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


def qaplotABC(run, T, sumstat=['smf'], nsnap0=15, sigma_smhm=0.2, downsampled='14', theta=None): 
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
    # data 
    m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_sim['halo.m'][isSF], subcat_sim['m.sham'][isSF], 
            dmhalo=0.2, weights=subcat_dat['weights'][isSF])
    sub.plot(m_mid, sig_mhalo, c='k', ls='--', label='SHAM') 
    
    sig_dat = smhmr.sigma_logMstar(subcat_sim['halo.m'][isSF], subcat_sim['m.sham'][isSF], 
            weights=subcat_sim['weights'][isSF], dmhalo=0.2)
    sig_sim = smhmr.sigma_logMstar(subcat_sim['halo.m'][isSF], subcat_sim['m.star'][isSF], 
            weights=subcat_sim['weights'][isSF], dmhalo=0.2)

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
    ssfr_arr = Obvs.SSFR_SFMS(m_arr, UT.z_nsnap(1), theta_SFMS=subcat_sim['theta_sfms'])
    sub.plot(m_arr, ssfr_arr+m_arr+0.3, ls='--', c='k') 
    sub.plot(m_arr, ssfr_arr+m_arr-0.3, ls='--', c='k') 

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
            sfr_ms = Obvs.SSFR_SFMS(subcat_sim['m.star'][i_r], UT.z_nsnap(i_snap), theta_SFMS=subcat_sim['theta_sfms']) + \
                                    subcat_sim['m.star'][i_r], 
            dlogsfrs[:,0] =  sfr - sfr_ms 
        else: 
            sfr = subcat_sim['snapshot'+str(i_snap)+'_sfr'][i_r]
            sfr_ms = Obvs.SSFR_SFMS(subcat_sim['snapshot'+str(i_snap)+'_m.star'][i_r], UT.z_nsnap(i_snap), theta_SFMS=subcat_sim['theta_sfms']) + \
                    subcat_sim['snapshot'+str(i_snap)+'_m.star'][i_r]
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
    else: 
        plt.show() 
    plt.close()
    return None 

