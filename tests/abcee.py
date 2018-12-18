import time 
import numpy as np 
# -- centralms -- 
from centralms import util as UT
from centralms import sfh as SFH 
from centralms import abcee as ABC
from centralms import evolver as Evol
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


def testModel(run): 
    t0 = time.time() 
    shcat = ABC.model(run, [0.5, 0.4], nsnap0=15, downsampled='20', testing=True)
    print time.time() - t0 
    return None 


def plotABC(run, T, prior='flex'): 
    ''' Corner plots of ABC runs  
    '''
    # thetas
    abcout = ABC.readABC(run, T) 
    theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) 
            for i in range(len(abcout['theta'][0]))]
    theta_info = ABC.Theta(prior=prior) 
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
    
    fig = plt.figure(figsize=(5*(len(sumstat)+5),4))
    mbin = np.arange(8.1, 11.9, 0.1) - 2.*np.log10(0.7)
    for i_s, stat in enumerate(sumstat): 
        if stat == 'smf': 
            sub = fig.add_subplot(1, len(sumstat)+5, i_s+1)
            # Li-White SMF 
            marr, phi, phierr = Obvs.dataSMF(source='li-white')  
            phi *= (1. - np.array([Obvs.f_sat(mm, 0.05) for mm in marr])) # sallite fraction 

            # central SMF 
            phierr *= np.sqrt(1./(1.-np.array([Obvs.f_sat(mm, 0.05) for mm in marr])))
            #sub.errorbar(marr, phi, phierr, fmt='.k', label='Data')
            sub.fill_between(marr, phi-phierr, phi+phierr, color='k', alpha=0.5, linewidth=0, label='Data') 
            #sub.plot(0.5*(mbin[1:]+mbin[:-1]), sumdata[0], c='k', ls='--', label='Data')
            sub.plot(0.5*(mbin[1:]+mbin[:-1]), sumsim[0], c='C0', label='Sim.')# \n'+sim_lbl)
        
            # SF central SMF
            isSF = (subcat_sim['galtype'] == 'sf') 
            fsfs = np.clip(Evol.Fsfms(marr), 0., 1.) 
            fsfs_errscale = np.ones(len(marr))
            fsfs_errscale[fsfs < 1.] = np.sqrt(1./(1.-fsfs[fsfs < 1.]))
            sub.errorbar(marr, fsfs * phi, fsfs_errscale * phierr, fmt='.k') # data 
            mmm, smf_sf = Obvs.getMF(subcat_sim['m.star'][isSF], weights=subcat_sim['weights'][isSF])
            sub.plot(mmm, smf_sf, c='C0', ls='--')
            
            sub.set_xlabel('$log\;M_*$', fontsize=25)
            sub.set_xlim([9., 12.])
            sub.set_ylabel('$\Phi$', fontsize=25)
            sub.set_ylim([1e-6, 10**-1.75])
            sub.set_yscale('log')
            sub.legend(loc='lower left', fontsize=20) 
        else: 
            raise NotImplementedError

    # SHMR panel of SF galaxies  
    sub = fig.add_subplot(1,len(sumstat)+5,len(sumstat)+1) # SMHMR panel of SF galaxies  
    isSF = ((subcat_sim['galtype'] == 'sf') & (subcat_sim['weights'] > 0.)) # only SF galaxies 
    # SHAM SMHMR 
    DFM.hist2d(subcat_sim['m.max'][isSF], subcat_sim['m.sham'][isSF], 
            weights=subcat_sim['weights'][isSF], 
            levels=[0.68, 0.95], range=[[10., 14.], [9., 12.]], color='k', 
            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
    # model 
    DFM.hist2d(subcat_sim['m.max'][isSF], subcat_sim['m.star'][isSF], 
            weights=subcat_sim['weights'][isSF], 
            levels=[0.68, 0.95], range=[[10., 14.], [9., 12.]], color='C0', 
            plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub) 
    DFM.hist2d(subcat_sim['halo.m'][isSF], subcat_sim['m.star'][isSF], 
            weights=subcat_sim['weights'][isSF], 
            levels=[0.68, 0.95], range=[[10., 14.], [9., 12.]], color='C1', 
            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
    sub.plot([0.,0.], [0.,0.], c='k', lw=2, label='SHAM') 
    sub.plot([0.,0.], [0.,0.], c='C1', lw=2, label='model') 
    sub.legend(loc='lower right', fontsize=15) 
    sub.set_xlabel('$log\;M_{halo}$', fontsize=25)
    sub.set_xlim([10.5, 14]) 
    sub.set_ylabel('$log\,M_*$', fontsize=25)
    sub.set_ylim([9.0, 11.5]) 

    # scatter in the SHMR panel of SF galaxies  
    sub = fig.add_subplot(1,len(sumstat)+5,len(sumstat)+2) 
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
    sub = fig.add_subplot(1, len(sumstat)+5, len(sumstat)+3)
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
    sub = fig.add_subplot(1, len(sumstat)+5, len(sumstat)+4)
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
    
    sub = fig.add_subplot(1, len(sumstat)+5, len(sumstat)+5)
    mbins = np.linspace(9., 12., 20) 
    fq = np.zeros(len(mbins)-1)
    fq_sham = np.zeros(len(mbins)-1)
    for im in range(len(mbins)-1): 
        inmbin = ((subcat_sim['m.star'] > mbins[im]) & (subcat_sim['m.star'] < mbins[im+1]) & (subcat_sim['weights'] > 0))
        inmbin0 = ((subcat_sim['m.sham'] > mbins[im]) & (subcat_sim['m.sham'] < mbins[im+1]) & (subcat_sim['weights'] > 0))
        if np.sum(inmbin) > 0:  
            fq[im] = np.sum(subcat_sim['weights'][inmbin & (subcat_sim['galtype'] == 'sf')])/np.sum(subcat_sim['weights'][inmbin])
        if np.sum(inmbin0) > 0: 
            fq_sham[im] = np.sum(subcat_sim['weights'][inmbin0 & (subcat_sim['galtype'] == 'sf')])/np.sum(subcat_sim['weights'][inmbin0])
    
    sub.plot(0.5*(mbins[1:] + mbins[:-1]), 1.-fq, c='C1')  
    sub.plot(0.5*(mbins[1:] + mbins[:-1]), 1.-fq_sham, c='k', ls='--')  
    sub.set_xlim([9., 12.]) 
    sub.set_ylim([0., 1.]) 

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


def qaplotABC_Mhacc(run, T, sumstat=['smf'], nsnap0=15, sigma_smhm=0.2, downsampled='20', theta=None, figure=None): 
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
    
    tt = ABC._model_theta(run, theta_med)
    # galaxies with log Mh~11 but log M*~10.5
    outlier = ((subcat_sim['galtype'] == 'sf') & (subcat_sim['weights'] > 0.) & 
            (subcat_sim['halo.m'] > 11.0) & (subcat_sim['halo.m'] < 11.1) & 
            (subcat_sim['m.star'] > 10.2) & 
            (subcat_sim['nsnap_start'] == 15)) 
    control = ((subcat_sim['galtype'] == 'sf') & (subcat_sim['weights'] > 0.) & 
            (subcat_sim['halo.m'] > 11.0) & (subcat_sim['halo.m'] < 11.1) & 
            (subcat_sim['m.star'] < 10.2) & 
            (subcat_sim['nsnap_start'] == 15)) 
    print np.sum(outlier), np.sum(control) 
    n_halo = len(subcat_sim['m.star'])
    i_outlier = np.random.choice(np.arange(n_halo)[outlier], 2, replace=False) 
    i_control = np.random.choice(np.arange(n_halo)[control], np.sum(control), replace=False) 

    # get accretion history 
    mhacc_outlier = np.zeros((len(i_outlier), 15))
    mhacc_outlier[:,0] = subcat_sim['halo.m'][i_outlier]
    msacc_outlier = np.zeros((len(i_outlier), 15))
    msacc_outlier[:,0] = subcat_sim['m.star'][i_outlier]
    dsfrs_outlier = np.zeros((len(i_outlier), 15))
    dsfrs_outlier[:,0] = subcat_sim['sfr'][i_outlier] - SFH.SFR_sfms(subcat_sim['m.star'][i_outlier], UT.z_nsnap(1), tt['sfms'])
    for i in range(2, 16): 
        mhacc_outlier[:,i-1] = subcat_sim['halo.m.snap'+str(i)][i_outlier]
        if i != 15: 
            msacc_outlier[:,i-1] = subcat_sim['m.star.snap'+str(i)][i_outlier]
            dsfrs_outlier[:,i-1] = subcat_sim['sfr.snap'+str(i)][i_outlier]
        else: 
            msacc_outlier[:,i-1] = subcat_sim['m.star0'][i_outlier]
            dsfrs_outlier[:,i-1] = subcat_sim['sfr0'][i_outlier]
        
    mhacc_control = np.zeros((len(i_control), 15))
    mhacc_control[:,0] = subcat_sim['halo.m'][i_control]
    msacc_control = np.zeros((len(i_control), 15))
    msacc_control[:,0] = subcat_sim['m.star'][i_control]
    dsfrs_control = np.zeros((len(i_control), 15))
    dsfrs_control[:,0] = subcat_sim['sfr'][i_control] - SFH.SFR_sfms(subcat_sim['m.star'][i_control], UT.z_nsnap(1), tt['sfms'])
    for i in range(2, 16):  
        mhacc_control[:,i-1] = subcat_sim['halo.m.snap'+str(i)][i_control]
        if i != 15: 
            msacc_control[:,i-1] = subcat_sim['m.star.snap'+str(i)][i_control]
            dsfrs_control[:,i-1] = subcat_sim['sfr.snap'+str(i)][i_control]
        else:
            msacc_control[:,i-1] = subcat_sim['m.star0'][i_control]
            dsfrs_control[:,i-1] = subcat_sim['sfr0'][i_control]
        
    fig = plt.figure(figsize=(15,4))
    sub = fig.add_subplot(131) 
    #sub.plot([1., 0.], [1., 1.], c='k', ls='--') 
    for i in range(len(i_outlier)): 
        sub.plot(UT.z_nsnap(np.arange(1, 16)[::-1]), 10**(mhacc_outlier[i,:][::-1]), c='C'+str(i))
    for i in range(len(i_control)): 
        sub.plot(UT.z_nsnap(np.arange(1, 16)[::-1]), 10**(mhacc_control[i,:][::-1]), c='k', lw=0.1)
    sub.set_xlabel('$z$ (redshift)', fontsize=25)
    sub.set_xlim([1., 0.])
    sub.set_yscale('log') 
    sub.set_ylabel('$M_h$', fontsize=25)
    sub.set_ylim([5e10, 5e11])
    #sub.set_ylabel('$M_h/M_h(z=0)$', fontsize=25)
    #sub.set_ylim([0., 1.2])
    sub.text(0.95, 0.05, r'$\log\,M_h(z=0) = 11$', fontsize=20, ha='right', va='bottom', transform=sub.transAxes)
    
    sub = fig.add_subplot(132) 
    for i in range(len(i_outlier)): 
        sub.plot(UT.z_nsnap(np.arange(1, 16)[::-1]), 10**(msacc_outlier[i,:][::-1]), c='C'+str(i))
    for i in range(len(i_control)): 
        sub.plot(UT.z_nsnap(np.arange(1, 16)[::-1]), 10**(msacc_control[i,:][::-1]), c='k', lw=0.1)
    sub.legend(loc='lower right', fontsize=15) 
    sub.set_xlabel('$z$ (redshift)', fontsize=25)
    sub.set_xlim([1., 0.])
    sub.set_yscale('log') 
    sub.set_ylabel('$M_*$', fontsize=25)
    #sub.set_ylabel('$M_h/M_h(z=0)$', fontsize=25)
    #sub.set_ylim([0., 1.2])

    sub = fig.add_subplot(133) 
    for i in range(len(i_outlier)): 
        sub.plot(UT.z_nsnap(np.arange(1, 16)[::-1]), (dsfrs_outlier[i,:][::-1]), c='C'+str(i))
    for i in range(len(i_control)): 
        sub.plot(UT.z_nsnap(np.arange(1, 16)[::-1]), (dsfrs_control[i,:][::-1]), c='k', lw=0.1)
    sub.legend(loc='lower right', fontsize=15) 
    sub.set_xlabel('$z$ (redshift)', fontsize=25)
    sub.set_xlim([1., 0.])
    #sub.set_yscale('log') 
    #sub.set_ylabel('$M_*$', fontsize=25)
    sub.set_ylabel('$\Delta \log\,\mathrm{SFR}$', fontsize=25)
    sub.set_ylim([-1., 1.])
    fig.subplots_adjust(wspace=0.3)
    fig_name = ''.join([UT.fig_dir(), 'qaplot.t', str(T), '.', run, '_Mhacc.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def shmr_slope(runs=['randomSFH0.5gyr.sfsmf.sfsbroken', 'rSFH_abias0.5_0.5gyr.sfsmf.sfsbroken', 'rSFH_abias0.99_0.5gyr.sfsmf.sfsbroken'], T=14, nsnap0=15, downsampled='20'):  
    smhmr = Obvs.Smhmr()
    fig = plt.figure(figsize=(15,4))
    sub = fig.add_subplot(131) 
    sub0 = fig.add_subplot(132) 
    sub1 = fig.add_subplot(133)
    for i_run, run in enumerate(runs): 
        abcout = ABC.readABC(run, T)
        # median theta 
        theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) for i in range(len(abcout['theta'][0]))]

        subcat_sim = ABC.model(run, theta_med, nsnap0=nsnap0, downsampled=downsampled) 
        tt = ABC._model_theta(run, theta_med)
    
        isSF = ((subcat_sim['galtype'] == 'sf') & (subcat_sim['weights'] > 0.)) # only SF galaxies 
    
        # simulation 
        m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_sim['halo.m'][isSF], subcat_sim['m.star'][isSF], 
                dmhalo=0.2, weights=subcat_sim['weights'][isSF])
        i0 = np.argmin(m_mid[m_mid > 12.] - 12.) 
        i1 = np.argmax(m_mid[m_mid < 12.] - 12.)
        print run
        print (mu_mhalo[i0] - mu_mhalo[i1])/(m_mid[i0] - m_mid[i1])
        sub.errorbar(m_mid+0.02*i_run, mu_mhalo, sig_mhalo, fmt='.C'+str(i_run), label=''.join(run.split('_')))
        sub0.plot(m_mid, mu_mhalo, c='C'+str(i_run)) 
        m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_sim['m.star'][isSF], subcat_sim['halo.m'][isSF], 
                dmhalo=0.2, weights=subcat_sim['weights'][isSF])
        sub1.errorbar(m_mid+0.02*i_run, mu_mhalo, sig_mhalo, fmt='.C'+str(i_run), label=''.join(run.split('_')))
    sub.legend(loc='lower right', handletextpad=0.1, fontsize=15) 
    sub.set_xlabel('$\log\,M_{halo}$', fontsize=25)
    sub.set_xlim([11., 14.])
    sub.set_ylabel('$\log\,M_*$', fontsize=25)
    sub.set_ylim([9., 11.5])
    sub0.set_xlabel('$\log\,M_{halo}$', fontsize=25)
    sub0.set_xlim([11., 14.])
    sub0.set_ylim([9., 11.5])
    sub1.set_xlabel('$\log\,M_*$', fontsize=25)
    sub1.set_xlim([9., 11.5])
    sub1.set_ylim([11., 14])
    fig_name = ''.join([UT.fig_dir(), 'smhr_comp.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def shmr_distribution(runs=['randomSFH0.5gyr.sfsmf.sfsbroken', 'rSFH_abias0.5_0.5gyr.sfsmf.sfsbroken', 'rSFH_abias0.99_0.5gyr.sfsmf.sfsbroken'], T=14, nsnap0=15, downsampled='20'):  
    smhmr = Obvs.Smhmr()
    fig = plt.figure(figsize=(10,4))
    sub = fig.add_subplot(121) 
    sub0 = fig.add_subplot(122) 
    for i_run, run in enumerate(runs): 
        abcout = ABC.readABC(run, T)
        # median theta 
        theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) for i in range(len(abcout['theta'][0]))]

        subcat_sim = ABC.model(run, theta_med, nsnap0=nsnap0, downsampled=downsampled) 
        tt = ABC._model_theta(run, theta_med)
    
        isSF = ((subcat_sim['galtype'] == 'sf') & (subcat_sim['weights'] > 0.)) # only SF galaxies 
   
        msbin = (isSF & (subcat_sim['m.star'] > 10.45) & (subcat_sim['m.star'] < 10.65))
        mhbin = (isSF & (subcat_sim['halo.m'] > 11.9) & (subcat_sim['halo.m'] < 12.1))

        _ = sub.hist(subcat_sim['halo.m'][msbin], weights=subcat_sim['weights'][msbin], histtype='step', 
                linewidth=2, color='C'+str(i_run), label=''.join(run.split('.sfsflex')[0].split('_')))
        _ = sub0.hist(subcat_sim['m.star'][mhbin], weights=subcat_sim['weights'][mhbin], histtype='step', 
                color='C'+str(i_run), linewidth=2) 
    
    sub.set_title('$10.35 < \log\,M_* < 10.65$', fontsize=20)
    sub.legend(loc='upper right', handletextpad=0.1, fontsize=10) 
    sub.set_xlabel('$\log\,M_{halo}$', fontsize=25)
    sub.set_xlim([11., 14.])
    sub0.set_title('$11.9 < \log\,M_h < 12.1$', fontsize=20)
    sub0.set_xlabel('$\log\,M_*$', fontsize=25)
    sub0.set_xlim([9., 11.5])
    fig_name = ''.join([UT.fig_dir(), 'smhrdist_comp.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


if __name__=="__main__": 
    #testModel()
    #qaplotABC('rSFH_abias0.5_5gyr.sfsflex', 12, theta=[0.5, 0.4], figure=''.join([UT.fig_dir(), 'evolvertest.png']))
    for tduty in ['0.5', '1', '2', '5']: #, '10']: 
        #plotABC('randomSFH'+tduty+'gyr.sfsflex', 14, prior='flex')
        #qaplotABC('randomSFH'+tduty+'gyr.sfsflex', 14)
        #plotABC('randomSFH'+tduty+'gyr.sfsmf.sfsflex', 14, prior='flex')
        #qaplotABC('randomSFH'+tduty+'gyr.sfsmf.sfsflex', 14)
        #plotABC('rSFH_abias0.5_'+tduty+'gyr.sfsflex', 14, prior='flex')
        #qaplotABC('rSFH_abias0.5_'+tduty+'gyr.sfsflex', 14)
        #plotABC('rSFH_abias0.99_'+tduty+'gyr.sfsflex', 14, prior='flex')
        #qaplotABC('rSFH_abias0.99_'+tduty+'gyr.sfsflex', 14)
        #plotABC('randomSFH'+tduty+'gyr.sfsmf.sfsbroken', 14, prior='broken')
        #qaplotABC('randomSFH'+tduty+'gyr.sfsmf.sfsbroken', 14)
        plotABC('rSFH_abias0.5_'+tduty+'gyr.sfsmf.sfsbroken', 14, prior='broken')
        qaplotABC('rSFH_abias0.5_'+tduty+'gyr.sfsmf.sfsbroken', 14)
        plotABC('rSFH_abias0.99_'+tduty+'gyr.sfsmf.sfsbroken', 14, prior='broken')
        qaplotABC('rSFH_abias0.99_'+tduty+'gyr.sfsmf.sfsbroken', 14)
    #testModel('rSFH_abias0.5_5gyr.sfsflex')
    #plotABC('nodutycycle.sfsmf.sfsbroken', 14, prior='broken')
    #qaplotABC('nodutycycle.sfsmf.sfsbroken', 14)
    
    #qaplotABC('randomSFH0.5gyr.sfsflex', 14)
    #qaplotABC('rSFH_abias0.5_0.5gyr.sfsflex', 14)
    #qaplotABC('rSFH_abias0.99_0.5gyr.sfsflex', 14)
    #qaplotABC_Mhacc('rSFH_abias0.99_0.5gyr.sfsflex', 14)
    shmr_slope()
    #shmr_distribution()
    # no duty cycle run
    #plotABC('nodutycycle.sfsflex', 14, prior='flex')
    #qaplotABC('nodutycycle.sfsflex', 14)
