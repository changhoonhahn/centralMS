'''


'''
import time
import numpy as np 

# -- local -- 
import env 
import abcee
import util as UT
import observables as Obvs
import corner as DFM 
import emcee

# --- plotting --- 
import matplotlib.pyplot as plt 
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


def test_SumData(): 
    ''' Make sure abcee.SumData returns something sensible with some hardcoded values 
     
    Takes roughly 0.7 seconds 
    '''
    t0 = time.time() 
    output = abcee.SumData(['smf'], nsnap0=15, downsampled='14') 
    print time.time() - t0 , ' seconds'
    return output


def test_SumSim(run):
    ''' Profile the simulation 

    Takes roughly ~5 seconds for "constant offset" 
    '''
    t0 = time.time() 
    # run the model 
    subcat = abcee.model(run, np.array([1.35, 0.6]), nsnap0=15, downsampled='14')
    # get summary statistics 
    output = abcee.SumSim(['smf'], subcat)
    print time.time() - t0, ' seconds'

    return output 


def test_SumSim_sigmaSMHM(run, sigma_smhm=0.2):
    '''  run model for different sigma_smhm 
    '''
    t0 = time.time() 
    # run the model 
    subcat = abcee.model(run, np.array([1.35, 0.6]), nsnap0=15, downsampled='14', 
            sigma_smhm=sigma_smhm)
    # get summary statistics 
    output = abcee.SumSim(['smf'], subcat)
    print time.time() - t0, ' seconds'

    return output 


def test_Dist(): 
    ''' Test distance metric 
    '''
    # data summary statistic
    sum_data = test_SumData()

    # simulation summary statistic 
    sum_sims = test_SumSim()

    # distance function  
    rho = abcee.roe_wrap(['smf'])  
    
    # calculate distance 
    d = rho(sum_sims, sum_data)
    
    return d 


def test_runABC(): 
    ''' Purely a test run to check that there aren't any errors 
        
    ################################
    Run successful!
    ################################
    '''
    abcee.runABC('test0', 2, [10.], N_p=5, sumstat=['smf'], nsnap0=15, downsampled='14')
    return None 


def test_readABC(T): 
    ''' Try reading in different ABC outputs and do basic plots 
    '''
    abcout = abcee.readABC('test0', T)
    
    # median theta 
    theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) for i in range(len(abcout['theta'][0]))]

    print theta_med


def test_ABCsumstat(run, T):#, sumstat=['smf']): 
    ''' Compare the summary statistics of the median T-th ABC particle pool with data.
    Hardcoded for smf only 
    '''
    # data summary statistic
    sumdata = abcee.SumData(['smf'], info=True, nsnap0=15)

    # median theta 
    abcout = abcee.readABC('test0', T)
    theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) for i in range(len(abcout['theta'][0]))]
    
    subcat = abcee.model(run, theta_med, nsnap0=15, downsampled='14')
    sumsim = abcee.SumSim(['smf'], subcat, info=True)

    fig = plt.figure()
    sub = fig.add_subplot(111)

    sub.plot(sumdata[0][0], sumdata[0][1], c='k', ls='--', label='Data')
    sub.plot(sumsim[0][0], sumsim[0][1], c='b', label='Sim.')

    sub.set_xlim([9., 12.])
    sub.set_xlabel('Stellar Masses $(\mathcal{M}_*)$', fontsize=25)
    sub.set_ylim([1e-6, 10**-1.75])
    sub.set_yscale('log')
    sub.set_ylabel('$\Phi$', fontsize=25)
    sub.legend(loc='upper right') 
    plt.show()

    return None 


def test_ABC_SMHMR(run, T):#, sumstat=['smf']): 
    ''' Compare the SMHMR the median T-th ABC particle pool with 'data'
    Hardcoded for smf only 
    '''
    # data summary statistic
    subcat_dat = abcee.Data(nsnap0=15)

    # median theta 
    abcout = abcee.readABC('test0', T)
    theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) for i in range(len(abcout['theta'][0]))]
    
    # F( median theta) 
    subcat_sim = abcee.model(run, theta_med, nsnap0=15, downsampled='14')

    fig = plt.figure()
    sub = fig.add_subplot(111)

    smhmr = Obvs.Smhmr()
    # simulation 
    m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_sim['m.max'], subcat_sim['m.star'], 
            dmhalo=0.2, weights=subcat_sim['weights'])
    sub.fill_between(m_mid, mu_mhalo - sig_mhalo, mu_mhalo + sig_mhalo, color='b', alpha=0.25, linewidth=0, edgecolor=None, 
            label='Sim.')
    # data 
    m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_dat['m.max'], subcat_dat['m.star'], weights=subcat_dat['weights'])
    sub.errorbar(m_mid, mu_mhalo, yerr=sig_mhalo, color='k', label='Data')

    sub.set_xlim([10., 15.])
    sub.set_xlabel('Halo Mass $(\mathcal{M}_{halo})$', fontsize=25)
    sub.set_ylim([8., 12.])
    sub.set_ylabel('Stellar Mass $(\mathcal{M}_*)$', fontsize=25)
    sub.legend(loc='upper right') 
    plt.show()

    return None 


def test_plotABC(run, T): 
    abcee.plotABC(run, T) 
    return None 


def test_qaplotABC(run, T): 
    abcee.qaplotABC(run,T)
    return None


def test_SFMS_highz(run, T, nsnap=15, lit='lee', nsnap0=15, downsampled='14'): 
    ''' Compare the best-fit SFMS parameters from ABC to literature at high z  
    '''
    if lit == 'lee': 
        lit_label = 'Lee et al. (2015)'
    # median ABC theta 
    abcout = abcee.readABC(run, T)
    theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) for i in range(len(abcout['theta'][0]))]

    subcat_sim = abcee.model(run, theta_med, nsnap0=nsnap0, downsampled=downsampled) 

    m_arr = np.arange(8., 12.1, 0.1)
    sfr_abc = Obvs.SSFR_SFMS(m_arr, UT.z_nsnap(nsnap), theta_SFMS=subcat_sim['theta_sfms']) + m_arr

    sfr_obv = Obvs.SSFR_SFMS_obvs(m_arr, UT.z_nsnap(nsnap), lit=lit) + m_arr
    
    fig = plt.figure()
    sub = fig.add_subplot(111)

    sub.fill_between(m_arr,  sfr_abc-0.3, sfr_abc+0.3, color='b', alpha=0.25, linewidth=0, edgecolor=None, 
            label=r'ABC $\theta_{median}$')
    sub.plot(m_arr, sfr_obv+0.3, ls='--', c='k', label=lit_label) 
    sub.plot(m_arr, sfr_obv-0.3, ls='--', c='k') 

    sub.set_xlim([8., 12.])
    sub.set_xlabel('$\mathtt{log\;M_*}$', fontsize=25)
    sub.set_ylim([-4., 3.])
    sub.set_ylabel('$\mathtt{log\;SFR}$', fontsize=25)
    sub.legend(loc='best') 
    fig.savefig(UT.fig_dir()+'SFMS.z'+str(round(UT.z_nsnap(nsnap),2))+'.'+run+'.'+lit+'.png', bbox_inches='tight')
    plt.close()
    return None 


def test_dMh_dMstar(run, theta, sigma_smhm=0.2, nsnap0=15, downsampled='14', flag=None): 
    '''
    '''
    subcat_sim = abcee.model(run, theta, 
            nsnap0=nsnap0, sigma_smhm=sigma_smhm, downsampled=downsampled) 

    isSF = np.where((subcat_sim['gclass'] == 'sf') & 
            (subcat_sim['weights'] > 0.) & 
            (subcat_sim['nsnap_start'] == nsnap0))[0] # only SF galaxies 
    assert subcat_sim['m.star'][isSF].min() > 0.
    #print subcat_sim['m.star'][isSF].min(), subcat_sim['m.star'][isSF].max() 
    #print subcat_sim['m.star0'][isSF].min(), subcat_sim['m.star0'][isSF].max() 

    dMh = np.log10(10**subcat_sim['halo.m'][isSF] - 10**subcat_sim['halo.m0'][isSF])
    dMstar = np.log10(10**subcat_sim['m.star'][isSF] - 10**subcat_sim['m.star0'][isSF])

    fig = plt.figure(figsize=(20,6)) 
    sub = fig.add_subplot(131)
    scat = sub.scatter(subcat_sim['halo.m0'][isSF], subcat_sim['halo.m'][isSF], 
            lw=0, s=5, c='k', cmap='hot') 
    sub.set_xlim([9.75, 14.5])
    sub.set_xlabel('$\mathtt{log\;M_h(z_0 \sim 1)}$', fontsize=20)
    sub.set_ylim([9.75, 14.5])
    sub.set_ylabel('$\mathtt{log\;M_h(z_f \sim 0)}$', fontsize=20)
    sub = fig.add_subplot(132)
    scat = sub.scatter(subcat_sim['m.star0'][isSF], subcat_sim['m.star'][isSF], 
            lw=0, s=5, c='k', cmap='hot') 
    sub.set_xlim([7., 12.])
    sub.set_xlabel('$\mathtt{log\;M_*(z_0 \sim 1)}$', fontsize=20)
    sub.set_ylim([7., 12.])
    sub.set_ylabel('$\mathtt{log\;M_*(z_f \sim 0)}$', fontsize=20)

    sub = fig.add_subplot(133)
    scat = sub.scatter(dMh, dMstar, lw=0, s=5, c=subcat_sim['halo.m'][isSF], cmap='hot', 
            vmin=10., vmax=14.5) 
    fig.colorbar(scat, ax=sub)
    sub.set_xlim([8., 14.5])
    sub.set_xlabel('$\mathtt{log(\; \Delta M_h\;)}$', fontsize=20)
    sub.set_ylim([9, 11.25])
    sub.set_ylabel('$\mathtt{log(\; \Delta M_*\;)}$', fontsize=20)

    if flag is None: 
        flag_str = ''
    else: 
        flag_str = '.'+flag 
    
    fig.savefig(''.join([UT.fig_dir(), 'dMh_dMstar.', run, '.sig_smhm', str(sigma_smhm), 
        flag_str, '.png']), bbox_inches='tight')
    plt.close() 

    return None 


def test_tdelay_dt_grid(run, theta, sigma_smhm=0.2, nsnap0=15, downsampled='14', flag=None): 
    ''' plot sigma_M* on a grid of t_delay and dt 
    '''
    tdelays = np.arange(0., 3., 0.5)
    dts = np.arange(0., 4.5, 0.5)
    dts[0] += 0.1 # dt = 0 not allowed

    smhmr = Obvs.Smhmr()
    sig_Mstars = np.zeros((len(tdelays), len(dts)))
    #grid of tdelay, dt
    for i_t, tdelay in enumerate(tdelays): 
        for i_d, dt in enumerate(dts): 
            theta_i = np.concatenate([theta, np.array([tdelay, dt])])
            subcat_sim = abcee.model(run, theta_i, 
                    nsnap0=nsnap0, sigma_smhm=sigma_smhm, downsampled=downsampled) 
            sumsim = abcee.SumSim(['smf'], subcat_sim, info=True)
            
            isSF = np.where(subcat_sim['gclass'] == 'sf') # only SF galaxies 
            # calculate sigma_M* at M_h = 12
            m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_sim['halo.m'][isSF], subcat_sim['m.star'][isSF], 
                    dmhalo=0.2, weights=subcat_sim['weights'][isSF])
            sig_Mstars[i_t, i_d] = sig_mhalo[np.argmin(np.abs(m_mid-12.))]
            print 'tdelay = ', tdelay, ', dt = ', dt, ', sigma = ', sig_Mstars[i_t, i_d]

    fig = plt.figure() 
    sub = fig.add_subplot(111)
    im = plt.imshow(sig_Mstars, interpolation='None', cmap='hot', extent=(dts.min(), dts.max(), tdelays.min(), tdelays.max()))
    plt.colorbar(im)
    sub.set_xticks(dts) 
    sub.set_xlabel('$\Delta t$ Gyr', fontsize=25)
    sub.set_yticks(tdelays) 
    sub.set_ylabel('$t_{delay}$ Gyr', fontsize=25)

    fig_name = ''.join([UT.fig_dir(), run, '.tdelay_dt_grid.png'])
    fig.savefig(fig_name, bbox_inches='tight') 
    return None


def test_tdelay_dt_mcmc(run):
    '''
    '''
    chain_file = ''.join([UT.fig_dir(), run, '.tdelay_dt_mcmc.chain.dat']) 
    samples = np.loadtxt(chain_file) 
    
    #samples = sampler.chain[:, 5:, :].reshape((-1, Ndim))
    fig = corner.corner(samples, labels=['$t_{delay}$', '$\Delta t_{bias}$'])
    fig.savefig(UT.fig_dir()+run+'.tdelay_dt_mcmc.png')
    return None 


if __name__=='__main__': 
    #test_SFMS_highz('test0', 9, nsnap=15, lit='lee')

    #test_SumSim('rSFH_r1.0_most')
    #test_SumSim_sigmaSMHM('rSFH_r1.0_most', sigma_smhm=0.0)
    #abcee.qaplotABC('randomSFH_short', 10, sigma_smhm=0.0, theta=np.array([1.35, 0.6])) 
    #test_dMh_dMstar('test0', np.array([1.35, 0.6]), sigma_smhm=0.2)
    #test_dMh_dMstar('randomSFH_short', np.array([1.35, 0.6]), sigma_smhm=0.2)
    #test_dMh_dMstar('randomSFH_r0.99', np.array([1.35, 0.6]), sigma_smhm=0.2)
    #for t in np.arange(0.1, 4.5, 0.5): 
        #test_dMh_dMstar('rSFH_r0.99_delay_dt_test', np.array([1.35, 0.6, t]), sigma_smhm=0.2, flag='dt'+str(t)+'gyr')
    #    abcee.qaplotABC('rSFH_r0.99_delay_dt_test', 10, sigma_smhm=0.2, theta=np.array([1.35, 0.6, t]), figure=UT.fig_dir()+'rSFH_r0.99.delay0.dt'+str(t)+'.png') 
    #test_tdelay_dt_grid('rSFH_r0.99_delay_dt_test', np.array([1.35, 0.6]), sigma_smhm=0.2)
    test_tdelay_dt_mcmc('rSFH_r0.99_delay_dt_test')

    #abcee.qaplotABC('rSFH_r0.99_delay_dt_test', 10, sigma_smhm=0.2, theta=np.array([1.35, 0.6, 2.]), figure=UT.fig_dir()+'testing.dMmax.png') 
    #abcee.qaplotABC('randomSFH', 10, sigma_smhm=0.0, theta=np.array([1.35, 0.6])) 
    #abcee.qaplotABC('randomSFH_long', 10, sigma_smhm=0.0, theta=np.array([1.35, 0.6])) 
    #test_plotABC('randomSFH', 1)
    #test_qaplotABC('test0', 9)
    #test_ABCsumstat('randomSFH', 7)
    #sfh_name = 'rSFH_r0.99_delay'
    #sfh_name = 'randomSFH_short'
    #for t in [9]: #range(10)[::-1]: #[5,6]: #range(5):
    #    #test_plotABC(sfh_name, t)
    #    test_qaplotABC(sfh_name, t)
