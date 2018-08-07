import numpy as np 
# -- centralms -- 
from centralms import util as UT
from centralms import sfh as SFH 
from centralms import abcee as ABC
from centralms import observables as Obvs
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


def testModel(): 
    shcat = ABC.model('rSFH_abias0.99_5gyr.sfsflex', [0.5, 0.4], nsnap0=15, downsampled='20')
    return None 


def plotABC(run, T, prior='flex'): 
    ''' Corner plots of ABC runs  
    '''
    # thetas
    abcout = ABC.readABC(run, T) 
    theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) 
            for i in range(len(abcout['theta'][0]))]
    theta_info = ABC.Theta() 
    prior_obj = ABC.Prior(prior) # prior
    prior_range = [(prior_obj.min[i], prior_obj.max[i]) for i in range(len(prior_obj.min))]
    
    # figure name 
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
    
    fig_name = ''.join([UT.dat_dir(), 'abc/', run, '/', 't', str(T), '.', run , '.png'])
    fig.savefig(fig_name, bbox_inches="tight") 
    plt.close()
    return None 


def qaplotABC(run, T, sumstat=['smf'], nsnap0=15, sigma_smhm=0.2, downsampled='20', theta=None, figure=None): 
    ''' Quality assurance plot for ABC runs. Plot summary statistic(s), SMHMR, SFMS
    '''
    # first compare data summary statistics to Sim(median theta) 
    if theta is None:
        abcout = ABC.readABC(run, T)
        # median theta 
        theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) for i in range(len(abcout['theta'][0]))]
    else: 
        theta_med = theta
    
    # summary statistics of data 
    sumdata = ABC.dataSum(sumstat=sumstat)  
    # summary statistics of model 
    subcat_sim = ABC.model(run, theta_med, nsnap0=nsnap0, downsampled=downsampled) 
    sumsim = ABC.modelSum(subcat_sim, sumstat=sumstat)
    
    theta_info = ABC.Theta() 
    sim_lbl = ', \n'.join([ttt+'='+str(round(tm,2)) for ttt, tm in zip(theta_info['label'], theta_med)]) 
    
    fig = plt.figure(figsize=(5*(len(sumstat)+3),4))
    mbin = np.arange(8.1, 11.9, 0.1) - 2.*np.log10(0.7)
    for i_s, stat in enumerate(sumstat): 
        if stat == 'smf': 
            sub = fig.add_subplot(1, len(sumstat)+3, i_s+1)
            # Li-White SMF 
            marr, phi, phierr = Obvs.dataSMF(source='li-white')  
            phi *= (1. - np.array([Obvs.f_sat(mm, 0.05) for mm in marr])) # sallite fraction 
            phierr *= np.sqrt(1./(1.-np.array([Obvs.f_sat(mm, 0.05) for mm in marr])))
            #sub.errorbar(marr, phi, phierr, fmt='.k', label='Data')
            sub.fill_between(marr, phi-phierr, phi+phierr, color='k', alpha=0.5, linewidth=0, label='Data') 
            #sub.plot(0.5*(mbin[1:]+mbin[:-1]), sumdata[0], c='k', ls='--', label='Data')
            sub.plot(0.5*(mbin[1:]+mbin[:-1]), sumsim[0], c='C0', label='Sim. \n'+sim_lbl)
            sub.set_xlabel('$log\;M_*$', fontsize=25)
            sub.set_xlim([9., 12.])
            sub.set_ylabel('$\Phi$', fontsize=25)
            sub.set_ylim([1e-6, 10**-1.75])
            sub.set_yscale('log')
            sub.legend(loc='lower left', fontsize=20) 
        else: 
            raise NotImplementedError
    
    sub = fig.add_subplot(1,len(sumstat)+3,len(sumstat)+1) # SMHMR panel of SF galaxies  
    isSF = np.where(subcat_sim['galtype'] == 'sf') # only SF galaxies 
    smhmr = Obvs.Smhmr()
    # simulation 
    m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_sim['halo.m'][isSF], subcat_sim['m.star'][isSF], 
            dmhalo=0.2, weights=subcat_sim['weights'][isSF])
    enough = (cnts > 20) 
    sub.plot(m_mid[enough], sig_mhalo[enough], c='C0', lw=2, label='($M_{h}$)') 
    sig_sim = sig_mhalo[np.argmin(np.abs(m_mid-12.))]
    m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_sim['m.max'][isSF], subcat_sim['m.star'][isSF], 
            dmhalo=0.2, weights=subcat_sim['weights'][isSF])
    enough = (cnts > 20) 
    sub.plot(m_mid[enough], sig_mhalo[enough], c='#1F77B4', ls=':', lw=1, label='($M_{max}$)')
    # SHAM "data"
    m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_sim['halo.m'][isSF], subcat_sim['m.sham'][isSF], 
            dmhalo=0.2, weights=subcat_sim['weights'][isSF])
    enough = (cnts > 20) 
    sub.plot(m_mid[enough], sig_mhalo[enough], c='k', ls='--') 
    sig_dat = sig_mhalo[np.argmin(np.abs(m_mid-12.))]

    # mark sigma_M*(M_h = 10^12) 
    sub.text(0.95, 0.95, 
            ''.join(['$\sigma^{(s)}_{M_*}(M_h = 10^{12} M_\odot) = ', str(round(sig_sim,2)), '$ \n', 
                '$\sigma^{(d)}_{M_*}(M_h = 10^{12} M_\odot) = ', str(round(sig_dat,2)), '$']), 
            fontsize=15, ha='right', va='top', transform=sub.transAxes)
    sub.legend(loc='lower left', fontsize=15) 
    sub.set_xlabel('$log\;M_{halo}$', fontsize=25)
    sub.set_xlim([10.6, 15.])
    sub.set_ylabel('$\sigma_{log\,M_*}$', fontsize=25)
    sub.set_ylim([0., 0.6])

    # SFMS panel 
    tt = ABC._model_theta(run, theta_med)
    sub = fig.add_subplot(1, len(sumstat)+3, len(sumstat)+2)
    DFM.hist2d(
            subcat_sim['m.star'][isSF], 
            subcat_sim['sfr'][isSF], 
            weights=subcat_sim['weights'][isSF], 
            levels=[0.68, 0.95], range=[[8., 12.], [-4., 2.]], color='#1F77B4', 
            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
    sub.set_xlabel('$\log\;M_*$', fontsize=25)
    sub.set_xlim([8., 12.])
    sub.set_ylabel('$\log\;\mathrm{SFR}$', fontsize=25)
    sub.set_ylim([-4., 2.])
    
    # dSFR as a function of t_cosmic 
    sub = fig.add_subplot(1, len(sumstat)+3, len(sumstat)+3)
    mbins = np.linspace(9., 12., 10) 

    i_r = [] # select random SF galaxies over mass bins
    for i_m in range(len(mbins)-1): 
        inmbin = np.where(
                (subcat_sim['galtype'] == 'sf') & 
                (subcat_sim['nsnap_start'] == nsnap0) & 
                (subcat_sim['weights'] > 0) & 
                (subcat_sim['m.star'] > mbins[i_m]) & 
                (subcat_sim['m.star'] <= mbins[i_m+1]))
        if len(inmbin[0]) > 0: 
            i_r.append(np.random.choice(inmbin[0], size=1)[0])
    i_r = np.array(i_r)

    # calculate d(logSFR)  = logSFR - logSFR_MS 
    dlogsfrs = np.zeros((len(i_r), nsnap0-1))
    for i_snap in range(1, nsnap0): 
        snap_str = ''
        if i_snap != 1: snap_str = '.snap'+str(i_snap)

        sfr = subcat_sim['sfr'+snap_str][i_r]
        sfr_ms = SFH.SFR_sfms(subcat_sim['m.star'+snap_str][i_r], UT.z_nsnap(i_snap), tt['sfms']) 
        dlogsfrs[:,i_snap-1] =  sfr - sfr_ms 

    for i in range(dlogsfrs.shape[0]): 
        sub.plot(UT.t_nsnap(range(1, nsnap0))[::-1], dlogsfrs[i,:][::-1]) 
    for i in range(1, nsnap0): 
        sub.vlines(UT.t_nsnap(i), -1., 1., color='k', linestyle='--', linewidth=0.5)

    sub.set_xlim([UT.t_nsnap(nsnap0-1), UT.t_nsnap(1)])
    #sub.set_xticks([13., 12., 11., 10., 9.])
    sub.set_xlabel('$t_\mathrm{cosmic}$ [Gyr]', fontsize=25)
    sub.set_ylim([-1., 1.]) 
    sub.set_yticks([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9])
    sub.set_ylabel('$\Delta \log\,\mathrm{SFR}$', fontsize=25)

    fig.subplots_adjust(wspace=.3)
    if theta is None: 
        fig_name = ''.join([UT.dat_dir(), 'abc/', run, '/', 'qaplot.t', str(T), '.', run, '.png'])
        fig.savefig(fig_name, bbox_inches='tight')
        plt.close()
    else: 
        if figure is None: plt.show() 
        else: fig.savefig(figure, bbox_inches='tight')
        plt.close()
    return None 


if __name__=="__main__": 
    #testModel()
    #qaplotABC('rSFH_abias0.5_5gyr.sfsflex', 12, theta=[0.5, 0.4], figure=''.join([UT.fig_dir(), 'evolvertest.png']))
    for tduty in ['0.5', '1', '2', '5', '10']: 
        plotABC('rSFH_abias0.5_'+tduty+'gyr.sfsflex', 14, prior='flex')
        qaplotABC('rSFH_abias0.5_'+tduty+'gyr.sfsflex', 14)
        #plotABC('randomSFH'+tduty+'gyr.sfsflex', 14, prior='flex')
    #    qaplotABC('randomSFH'+tduty+'gyr.sfsflex', 14)
