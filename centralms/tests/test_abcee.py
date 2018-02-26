'''


'''
import time
import pickle
import numpy as np 

# -- local -- 
import env 
import abcee
import util as UT
import observables as Obvs
import emcee

# --- plotting --- 
import corner as DFM
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl 
import matplotlib.pyplot as plt 
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False
from ChangTools.plotting import prettycolors


def test_SumData(): 
    ''' Make sure abcee.SumData returns something sensible with some hardcoded values 
     
    Takes roughly 0.7 seconds 
    
    Notes
    -----
    * SumData returns sensible SMFs when compared to Li & White (2009) scaled by 
    1-f_sat from Wetzel et al.(2013).  
    '''
    t0 = time.time() 
    output = abcee.SumData(['smf'], info=True, nsnap0=15, sigma_smhm=0.2) 
    m_arr, phi_arr, err_arr = Obvs.MF_data()

    fcen = np.array([1. - Obvs.f_sat(mm, 0.05) for mm in m_arr]) 

    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.plot(output[0][0], output[0][1]) 
    sub.errorbar(m_arr, fcen * phi_arr, yerr=err_arr*np.sqrt(1./fcen)) 
    sub.set_xlim([9., 12.])
    sub.set_xlabel('$log\;M_*$', fontsize=25)
    sub.set_ylim([1e-6, 10**-1.75])
    sub.set_yscale('log')
    sub.set_ylabel('$\Phi$', fontsize=25)
    fig.savefig(''.join([UT.fig_dir(), 'tests/test.SumData.smf.png']), bbox_inches='tight') 
    return None 


def test_SumSim(run):
    ''' Profile the simulation 

    Takes roughly ~5 seconds for "constant offset" 
    '''
    t0 = time.time() 
    # run the model 
    subcat = abcee.model(run, np.array([1., -0.15]), nsnap0=15, downsampled='14')
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


def test_model_ABCparticle(run, T):  
    '''
    '''
    abcee.model_ABCparticle(run, T, nsnap0=15, sigma_smhm=0.2)
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

    fig = plt.figure(figsize=(26,6)) 
    sub = fig.add_subplot(141)
    scat = sub.scatter(subcat_sim['halo.m0'][isSF], subcat_sim['halo.m'][isSF], 
            lw=0, s=5, c='k', cmap='hot') 
    sub.set_xlim([9.75, 14.5])
    sub.set_xlabel('$\mathtt{log\;M_h(z_0 \sim 1)}$', fontsize=20)
    sub.set_ylim([9.75, 14.5])
    sub.set_ylabel('$\mathtt{log\;M_h(z_f \sim 0)}$', fontsize=20)
    sub = fig.add_subplot(142)
    scat = sub.scatter(subcat_sim['m.star0'][isSF], subcat_sim['m.star'][isSF], 
            lw=0, s=5, c='k', cmap='hot') 
    sub.set_xlim([7., 12.])
    sub.set_xlabel('$\mathtt{log\;M_*(z_0 \sim 1)}$', fontsize=20)
    sub.set_ylim([7., 12.])
    sub.set_ylabel('$\mathtt{log\;M_*(z_f \sim 0)}$', fontsize=20)

    sub = fig.add_subplot(143)
    scat = sub.scatter(dMh, dMstar, lw=0, s=5, c=subcat_sim['halo.m'][isSF], cmap='hot', 
            vmin=10., vmax=14.5) 
    fig.colorbar(scat, ax=sub)
    sub.set_xlim([8., 14.5])
    sub.set_xlabel('$\mathtt{log(\; \Delta M_h\;)}$', fontsize=20)
    sub.set_ylim([9, 11.25])
    sub.set_ylabel('$\mathtt{log(\; \Delta M_*\;)}$', fontsize=20)
    
    plt.show()
    raise ValueError
    if flag is None: 
        flag_str = ''
    else: 
        flag_str = '.'+flag 
    
    fig.savefig(''.join([UT.fig_dir(), 'dMh_dMstar.', run, '.sig_smhm', str(sigma_smhm), 
        flag_str, '.png']), bbox_inches='tight')
    plt.close() 
    return None 


def test_Mh_Mstar_assembly(sigma_smhm=0.2, nsnap0=15, downsampled='14'): 
    ''' Compare the assembly history of M_halo and M* for different amounts of assembly
    bias 
    '''
    thetas = [np.array([1.35, 0.6, rr, 0.5, 0., 0.5]) for rr in [0.1, 0.33, 0.66, 0.99]]
    
    fig = plt.figure(figsize=(6*len(thetas),6)) 
    for i_t, theta in enumerate(thetas): 
        subcat_sim = abcee.model('rSFH_r_delay_dt_test', theta, 
                nsnap0=nsnap0, sigma_smhm=sigma_smhm, downsampled=downsampled) 
        
        if i_t == 0: # pick a handful of halos 
            isSF = np.where((subcat_sim['gclass'] == 'sf') & 
                    (subcat_sim['weights'] > 0.) & 
                    (subcat_sim['nsnap_start'] == nsnap0) & 
                    (subcat_sim['halo.m'] > 13.) & 
                    (subcat_sim['halo.m'] < 14.))[0] # only SF galaxies 
            assert subcat_sim['m.star'][isSF].min() > 0.

            i_s = np.random.choice(isSF)#, size=10, replace=False) 
        
        Mhs, Mstars = np.zeros(nsnap0), np.zeros(nsnap0)
        Mhs[0] = subcat_sim['halo.m'][i_s]
        Mhs[-1]= subcat_sim['halo.m0'][i_s]
        Mstars[0] = subcat_sim['m.star'][i_s]
        Mstars[-1]= subcat_sim['m.star0'][i_s]
        for isnap in range(2,nsnap0):
            Mhs[isnap-1] = subcat_sim['snapshot'+str(isnap)+'_halo.m'][i_s]
            Mstars[isnap-1]= subcat_sim['snapshot'+str(isnap)+'_m.star'][i_s]

        if i_t == 0: # plot the halo accretion history 
            sub = fig.add_subplot(1,len(thetas), 1)
            
            f_Mh = (10**Mhs - 10**Mhs[-1])/(10**Mhs[0] - 10**Mhs[-1])
            #10**(Mhs - Mhs[0])

            sub.plot(UT.t_nsnap(range(1, nsnap0+1))[::-1], f_Mh[::-1])
            sub.set_xlim([6., 13.5])
            sub.set_xlabel('$\mathtt{t_{cosmic}}$', fontsize=20)
            sub.set_ylim([0., 1.])
            sub.set_ylabel('$\mathtt{f_{M_h} = M_h(t)/M_h(z=0)}$', fontsize=20)
            print Mhs[::-1]
        print 'R='+str(theta[2]), subcat_sim['weights'][i_s], subcat_sim['gclass'][i_s]
        print Mstars[::-1]

    
        #f_Mstar = 10**(Mstars - Mstars[0])
        f_Mstar = (10**Mstars - 10**Mstars[-1])/(10**Mstars[0] - 10**Mstars[-1])
        sub = fig.add_subplot(1,len(thetas)+1,i_t+2)
        #sub.plot(UT.t_nsnap(range(1, nsnap0+1))[::-1], f_Mstar[::-1])
        sub.plot(f_Mh[::-1], f_Mstar[::-1])
        #sub.set_xlim([6., 13.5])
        #sub.set_xlabel('$\mathtt{t_{cosmic}}$', fontsize=20)
        #sub.set_ylim([0., 1.])
        #sub.set_yticklabels([''])
        #sub.set_ylabel('$\mathtt{f_{M_*} = M_*(t)/M_*(z=0)}$', fontsize=20)
        sub.set_title('R='+str(theta[2]), fontsize=20)
    plt.show() 
    return None 


def tduty_tdelay_dt_grid(run, theta, sigma_smhm=0.2, nsnap0=15, downsampled='14', flag=None): 
    ''' plot sigma_M* on a grid of t_delay and dt 
    '''
    delt = 0.5
    tdelays = np.arange(0., 3., delt)
    dts = np.arange(0., 4.5, delt)
    dts[0] += 0.1 # dt = 0 not allowed
    tdutys = np.array([0.5, 1., 2., 3., 5., 10.])

    subcat_dat = abcee.Data(nsnap0=nsnap0, sigma_smhm=sigma_smhm) # 'data'
    sumdata = abcee.SumData(['smf'], nsnap0=nsnap0, sigma_smhm=sigma_smhm)  

    smhmr = Obvs.Smhmr()
    grid = np.zeros((5, len(tdutys)*len(tdelays)*len(dts)))
    ii = 0  
    for i_duty, tduty in enumerate(tdutys):
        for i_t, tdelay in enumerate(tdelays): 
            for i_d, dt in enumerate(dts): 
                theta_i = np.concatenate([theta, np.array([tduty, tdelay, dt])])
                #try: 
                subcat_sim = abcee.model(run, theta_i, 
                        nsnap0=nsnap0, sigma_smhm=sigma_smhm, downsampled=downsampled) 
                sumsim = abcee.SumSim(['smf'], subcat_sim)
                
                isSF = np.where(subcat_sim['gclass'] == 'sf') # only SF galaxies 
                # calculate sigma_M* at M_h = 12
                try: 
                    m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_sim['halo.m'][isSF], subcat_sim['m.star'][isSF], 
                            dmhalo=0.2, weights=subcat_sim['weights'][isSF])
                    grid[3,ii] = sig_mhalo[np.argmin(np.abs(m_mid-12.))]
                    # rho 
                    grid[4, ii] = abcee.L2_logSMF(sumsim, sumdata)
                except ValueError: 
                    grid[3,ii] = -999.
                    grid[4, ii] = 999. 
                print 'tduty = ', tduty, 'tdelay = ', tdelay, ', dt = ', dt, ', sigma = ', grid[3, ii]
                grid[0, ii] = tduty
                grid[1, ii] = tdelay
                grid[2, ii] = dt 
                ii += 1

    # save sig_Mstar values to file  
    file_name = ''.join([UT.fig_dir(), 'tduty_tdelay_dt_grid.', run, '.p'])
    pickle.dump(grid, open(file_name, 'wb'))
    return None 


def test_tduty_tdelay_dt_grid(run):
    '''
    '''
    file_name = ''.join([UT.fig_dir(), 'tduty_tdelay_dt_grid.', run, '.p'])
    grid = pickle.load(open(file_name, 'rb'))

    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(111, projection='3d')
    #sub.view_init(45, 60)

    tduty = grid[0,:]
    tdelay = grid[1,:] 
    dt_abias = grid[2,:]
    sigMstar = grid[3,:]
    l2 = grid[4,:]
    print 'L2 = ', l2.min(), l2.max() 

    scat = sub.scatter(dt_abias, tdelay, sigMstar, c=tduty)#, facecolors=cm.viridis(tduty))
    sub.set_xlabel('$\Delta t_{abias}$ Gyr', fontsize=15)
    sub.set_ylabel('$t_{delay}$ Gyr', fontsize=15)
    sub.set_zlabel('$\sigma_{log\,M_*}$ Gyr', fontsize=15)
    fig.colorbar(scat, shrink=0.5, aspect=10, cmap='hot', label='$t_{duty}$ Gyr')

    for angle in range(0, 360):
        sub.view_init(10, angle)
        fig_name = ''.join([UT.fig_dir(), 'tduty_tdelay_dt_grid.', run, '.', str(angle), '.png'])
        fig.savefig(fig_name)
    return None 


def test_tduty_tdelay_dt_grid_best(run):
    '''
    '''
    file_name = ''.join([UT.fig_dir(), 'tduty_tdelay_dt_grid.', run, '.p'])
    grid = pickle.load(open(file_name, 'rb'))
    
    tduty = grid[0,:]
    tdelay = grid[1,:] 
    dt_abias = grid[2,:]
    sigMstar = grid[3,:]
    l2 = grid[4,:]

    lim = np.where(tdelay > 1.)
    
    i_min = np.argmin(sigMstar[lim])
    i_min = lim[0][i_min]
    print grid[:,i_min]

    abcee.qaplotABC(run, 10, sigma_smhm=0.2, theta=np.array([1.35, 0.6, grid[0,i_min], grid[1,i_min], grid[2,i_min]])) 
    return None


def test_tdelay_dt_grid(run, tduty):
    file_name = ''.join([UT.fig_dir(), 'tduty_tdelay_dt_grid.', run, '.p'])
    grid = pickle.load(open(file_name, 'rb'))
    
    tdutys = grid[0,:]
    tdelay = grid[1,:] 
    dt_abias = grid[2,:]
    sigMstar = grid[3,:]
    l2 = grid[4,:]
    
    duty = np.where(tdutys == tduty)

    fig = plt.figure() 
    sub = fig.add_subplot(111)
    scat = sub.scatter(dt_abias[duty], tdelay[duty], c=sigMstar[duty], lw=0, s=100)
    plt.colorbar(scat)
    sub.set_xlabel('$\Delta t$ Gyr', fontsize=25)
    sub.set_ylabel('$t_{delay}$ Gyr', fontsize=25)

    fig_name = ''.join([UT.fig_dir(), 'tdelay_dt_grid.', run, '.tduty', str(tduty), '.png'])
    fig.savefig(fig_name, bbox_inches='tight') 
    return None


def test_Mh_SFR(run, theta, nsnap0=15, sigma_smhm=0.2, downsampled='14'):
    ''' Take a look at the M_h SFR relationship 
    '''
    subcat_sim = abcee.model(run, theta, 
            nsnap0=nsnap0, sigma_smhm=sigma_smhm, downsampled=downsampled) 

    isSF = np.where(subcat_sim['gclass'] == 'sf') 

    fig = plt.figure()
    sub = fig.add_subplot(111)
    DFM.hist2d(subcat_sim['halo.m'][isSF], subcat_sim['sfr'][isSF], weights=subcat_sim['weights'][isSF], 
            levels=[0.68, 0.95], range=[[10., 15.], [-4., 2.]], color='#1F77B4', 
            plot_datapoints=True, fill_contours=False, plot_density=False, ax=sub) 
    plt.show() 


def test_tduty_tdelay_dt_grid_dtaxis(run, tdutyy, tdelayy):
    ''' Check the dependence of sigma_logM* to dt_abias. It's weird that 
    dt_abias has such a significant effect... 
    '''
    file_name = ''.join([UT.fig_dir(), 'tduty_tdelay_dt_grid.', run, '.p'])
    grid = pickle.load(open(file_name, 'rb'))
    
    tduty = grid[0,:]
    tdelay = grid[1,:] 
    dt_abias = grid[2,:]
    sigMstar = grid[3,:]
    l2 = grid[4,:]

    lim = np.where((tdelay == tdelayy) & (tduty == tdutyy))
        
    for ii in range(len(lim[0])): 
        print dt_abias[lim[0][ii]], sigMstar[lim[0][ii]]

    #abcee.qaplotABC(run, 10, sigma_smhm=0.2, theta=np.array([1.35, 0.6, grid[0,i_min], grid[1,i_min], grid[2,i_min]])) 
    return None


if __name__=='__main__': 
    #test_SFMS_highz('test0', 9, nsnap=15, lit='lee')

    #test_SumSim('rSFH_r0.99_tdyn_0.5Gyr')#'rSFH_r1.0_most')
    #test_SumSim_sigmaSMHM('rSFH_r1.0_most', sigma_smhm=0.0)
    #for rr in [0.1, 0.33, 0.66, 0.99]: 
    #    test_dMh_dMstar('rSFH_r_delay_dt_test', np.array([1.35, 0.6, rr, 0.5, 0., 0.5]), sigma_smhm=0.2, flag='R'+str(rr))
    #abcee.qaplotABC('rSFH_r_delay_dt_test', 10, sigma_smhm=0.2, theta=np.array([1.35, 0.6, 0.99, 10., 0., 1.])) 

    #for t in np.arange(0.1, 4.5, 0.5): 
        #test_dMh_dMstar('rSFH_r0.99_delay_dt_test', np.array([1.35, 0.6, t]), sigma_smhm=0.2, flag='dt'+str(t)+'gyr')
    #    abcee.qaplotABC('rSFH_r0.99_delay_dt_test', 10, sigma_smhm=0.2, theta=np.array([1.35, 0.6, t]), figure=UT.fig_dir()+'rSFH_r0.99.delay0.dt'+str(t)+'.png') 
    #tduty_tdelay_dt_grid('rSFH_r0.99_delay_dt_test', np.array([1.35, 0.6]), sigma_smhm=0.2)
    #test_tdelay_dt_grid('rSFH_r0.99_delay_dt_test', 0.5)
    #test_tduty_tdelay_dt_grid_best('rSFH_r0.99_delay_dt_test')
    #test_tduty_tdelay_dt_grid_dtaxis('rSFH_r0.99_delay_dt_test', 1., 0.)
    #test_Mh_SFR('rSFH_r_delay_dt_test', np.array([1.35, 0.6, 0.99, 10., 0., 1.])) 

    #abcee.qaplotABC('rSFH_r0.99_delay_dt_test', 10, sigma_smhm=0.2, theta=np.array([1.35, 0.6, 2.]), figure=UT.fig_dir()+'testing.dMmax.png') 
    #abcee.qaplotABC('randomSFH', 10, sigma_smhm=0.0, theta=np.array([1.35, 0.6])) 
    #abcee.qaplotABC('randomSFH_long', 10, sigma_smhm=0.0, theta=np.array([1.35, 0.6])) 
    #test_plotABC('randomSFH', 1)
    #test_qaplotABC('test0', 9)
    #test_ABCsumstat('randomSFH', 7)
    #sfh_name = 'rSFH_r0.99_delay'
    #sfh_name = 'test0' # 'randomSFH_short'

    sfh_name = 'rSFH_r0.99_tdyn_0.5gyr' # 0.5gyr_narrSFMS'#'randomSFH_5gyr' # 'randomSFH_short'
    #sfh_name = 'randomSFH_0.5gyr'#_narrSFMS' # 'randomSFH_short'
    #sfh_name = 'rSFH_r0.5_tdyn_0.5gyr'
    #test_model_ABCparticle(sfh_name, 13)
    for t in [14]: #range(1,12)[::-1]: #[5,6]: #range(5):
        test_plotABC(sfh_name, t)
        test_qaplotABC(sfh_name, t)
