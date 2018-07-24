import numpy as np 
# -- centralms -- 
from centralms import util as UT
from centralms import abcee as ABC
# -- plotting --
import corner as DFM 
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


def plotABC(run, T): 
    ''' Corner plots of ABC runs  
    '''
    # thetas
    abcout = ABC.readABC(run, T) 
    theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) 
            for i in range(len(abcout['theta'][0]))]
    theta_info = ABC.Theta() 
    prior_obj = ABC.Prior('anchored') # prior
    prior_range = [(prior_obj.min[i], prior_obj.max[i]) for i in range(len(prior_obj.min))]
    
    # figure name 
    fig_name = ''.join([UT.dat_dir(), 'abc/', run, '/', 't', str(T), '.', run , '.png'])
    
    fig = DFM.corner(
            abcout['theta'],                            # thetas
            weights=abcout['w'].flatten(),              # weights
            truths=theta_med, truth_color='#ee6a50',    # median theta 
            labels=theta_info['label'], 
            label_kwargs={'fontsize': 25},
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
    enough = (cnts > 50) 
    sub.plot(m_mid[enough], sig_mhalo[enough], c='#1F77B4', lw=2, label='Model') 
    sig_sim = sig_mhalo[np.argmin(np.abs(m_mid-12.))]
    m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_sim['m.max'][isSF], subcat_sim['m.star'][isSF], 
            dmhalo=0.2, weights=subcat_sim['weights'][isSF])
    enough = (cnts > 50) 
    sub.plot(m_mid[enough], sig_mhalo[enough], c='#1F77B4', ls='--', lw=2)#, label='Model') 
    # data 
    m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_sim['halo.m'][isSF], subcat_sim['m.sham'][isSF], 
            dmhalo=0.2, weights=subcat_dat['weights'][isSF])
    enough = (cnts > 50) 
    sub.plot(m_mid[enough], sig_mhalo[enough], c='k', ls='--', label='SHAM') 
    sig_dat = sig_mhalo[np.argmin(np.abs(m_mid-12.))]
    m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_sim['m.max'][isSF], subcat_sim['m.sham'][isSF], 
            dmhalo=0.2, weights=subcat_dat['weights'][isSF])
    enough = (cnts > 50) 
    sub.plot(m_mid[enough], sig_mhalo[enough], c='k', ls=':') 
    
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


if __name__=="__main__": 
    plotABC('randomSFH_5gy', 4)
