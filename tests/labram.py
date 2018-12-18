'''
'''
import numpy as np 

from centralms import util as UT
from centralms import labram as LA2016
from centralms import catalog as Cat
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


def LAbramCatalog(): 
    '''
    '''
    cat = LA2016.catalogAbramson2016() 

    fig = plt.figure(figsize=(6., 5.))
    sub = fig.add_subplot(111)
    i_z = np.argmin(np.abs(cat['REDSHIFT'] - 1.)) #  index that best match to z~1 
    sub.scatter(np.log10(cat['MSTEL_T'][:,i_z]), np.log10(cat['SFR_T'][:,i_z]), c='C0', s=2, label='$z\sim1$') 
    sub.scatter(np.log10(cat['MSTEL_OBS']), np.log10(cat['SFR_OBS']), c='k', s=2, label='$z=z_\mathrm{obs}$') 
    sub.set_xlim([9., 12.]) 
    sub.set_xlabel('$\log\;M_*$', fontsize=25)
    sub.set_ylim([-2., 2.]) 
    sub.set_ylabel('$\log\;\mathrm{SFR}$', fontsize=25)
    sub.legend(loc='lower left', markerscale=10, handletextpad=0., fontsize=20)
    
    ffig = ''.join([UT.fig_dir(), 'catalogAbramson.png'])
    fig.savefig(ffig, bbox_inches="tight") 
    plt.close()
    return None 


def matchSMF(nsnap=15, n_boot=20): 
    ''' test that LA2016.matchSMF weights reproduces SMFs
    '''
    cat = LA2016.catalogAbramson2016() 
    i_z = np.argmin(np.abs(cat['REDSHIFT'] - UT.z_nsnap(nsnap))) #  index that best match to nsnapshot 15 
    mstar = np.log10(cat['MSTEL_T'][:,i_z])
    ws = LA2016.matchSMF(mstar, UT.z_nsnap(nsnap), logM_range=[9., 12.]) # get the weights to match SMF
    
    # analytic SMF 
    mbin = np.arange(9.0, 12.1, 0.1)
    phi = Obvs.analyticSMF(UT.z_nsnap(nsnap), m_arr=mbin, dlogm=0.1, source='li-march')
    
    mbin = np.arange(9.0, 12.2, 0.2)
    bbb, wgal = Obvs.getMF(mstar, weights=ws, mbin=mbin) #np.histogram(mstar, weights=ws, bins=mbin)
    wgal_boots = np.zeros((n_boot, len(wgal)))
    for ib in range(n_boot): 
        iboot = np.random.choice(np.arange(len(mstar)), size=len(mstar), replace=True) 
        _, wgal_iboot = Obvs.getMF(mstar[iboot], weights=ws[iboot], mbin=mbin)# np.histogram(mstar[iboot], weights=ws[iboot], bins=mbin)
        wgal_boots[ib,:] = wgal_iboot
    sig_wgal = np.std(wgal_boots, axis=0)
    
    fig = plt.figure(figsize=(5.,5.))
    sub = fig.add_subplot(111)
    sub.plot(phi[0], phi[1], c='C0') # anlaytic
    sub.errorbar(bbb, wgal, sig_wgal, fmt='.k')
    sub.set_xlim([9., 12.])
    sub.set_xlabel('$log\;M_*$', fontsize=25)
    sub.set_ylim([1e-5, 3e-2])
    sub.set_ylabel('$\Phi$', fontsize=25)
    sub.set_yscale("log")

    ffig = ''.join([UT.fig_dir(), 'catalogAbramson.SMFnsnap', str(nsnap), '.png'])
    fig.savefig(ffig, bbox_inches="tight") 
    plt.close()
    return None


def assignHalo(nsnap=15): 
    # read in subhalo catalog
    cat = Cat.CentralSubhalos(sigma_smhm=0.2, smf_source='li-march', nsnap0=15) 
    sh = cat.Read()
    # read in Louis's catalog
    cat = LA2016.catalogAbramson2016() 
    i_z = np.argmin(np.abs(cat['REDSHIFT'] - UT.z_nsnap(nsnap))) #  index that best match to snapshot 
    mstar = np.log10(cat['MSTEL_T'][:,i_z]) # M* @ z ~ zsnap
    ws = LA2016.matchSMF(mstar, UT.z_nsnap(nsnap), logM_range=[9., 12.]) # get the weights to match SMF
    
    # assign Mhalo, Mmax to Louis's catalog
    mhalo, mmax, i_halo = LA2016.assignHalo(mstar, nsnap, sh, logM_range=[9., 12.], dlogMs=0.2) 
    
    # now lets compare the resulting SHMRs 
    fig = plt.figure(figsize=(20,5))
    # first a scatter plot
    sub = fig.add_subplot(141) 
    sub.scatter(sh['halo.m.snap'+str(nsnap)], sh['m.sham.snap'+str(nsnap)], c='k', s=1, label='TreePM SHAM') 
    sub.scatter(mhalo, mstar, c='C1', s=2, label='weighted Abramson+(2016)') 
    sub.set_xlabel('log\,$M_h$', fontsize=25) 
    sub.set_xlim([11., 14.]) 
    sub.set_ylabel('log\,$M_*$', fontsize=25) 
    sub.set_ylim([9., 12.]) 
    sub.legend(loc=(0., 0.8), markerscale=5, handletextpad=0., frameon=True, fontsize=15) 

    # a DFM contour plot
    sub = fig.add_subplot(142)
    DFM.hist2d(sh['halo.m.snap'+str(nsnap)], sh['m.sham.snap'+str(nsnap)], 
            levels=[0.68, 0.95], range=[[11., 15.], [9., 12.]], color='k', 
            bins=20, plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub) 
    DFM.hist2d(mhalo, mstar, weights=ws, 
            levels=[0.68, 0.95], range=[[11., 15.], [9., 12.]], color='C1', 
            bins=20, plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub) 
    sub.set_xlabel('log\,$M_h$', fontsize=25) 
    sub.set_xlim([11., 14.]) 
    sub.set_ylim([9., 12.]) 

    # SHMR plot
    smhmr = Obvs.Smhmr()
    sub = fig.add_subplot(143)
    mbin = np.arange(11., 15., 0.2) 
    ms_nonzero = (sh['m.sham.snap'+str(nsnap)] > 0.) 
    m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(sh['halo.m.snap'+str(nsnap)][ms_nonzero], 
            sh['m.sham.snap'+str(nsnap)][ms_nonzero], m_bin=mbin)
    sub.errorbar(m_mid, mu_logMs, sig_logMs, fmt='.k') 
    m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(mhalo, mstar, weights=ws, m_bin=mbin)
    sub.errorbar(m_mid[m_mid > 11.8], mu_logMs[m_mid > 11.8], sig_logMs[m_mid > 11.8], fmt='.C1') 
    sub.set_xlabel('log $M_h$', fontsize=25) 
    sub.set_xlim([11., 14.]) 
    #sub.set_ylabel('log $M_*$', fontsize=25) 
    sub.set_ylim([9., 12.]) 
    #sub.set_ylim([0., 0.5]) 

    sub = fig.add_subplot(144)
    mbin = np.arange(9., 12., 0.2) 
    m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(sh['m.sham.snap'+str(nsnap)], sh['halo.m.snap'+str(nsnap)], m_bin=mbin)
    sub.errorbar(m_mid, mu_logMs, sig_logMs, fmt='.k') 
    m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(mstar, mhalo, weights=ws, m_bin=mbin)
    sub.errorbar(m_mid, mu_logMs, sig_logMs, fmt='.C1') 
    sub.set_xlabel('log $M_*$', fontsize=25) 
    sub.set_xlim([9., 12.]) 
    sub.set_ylabel('log $M_h$', fontsize=25) 
    sub.set_ylim([11., 14.]) 

    ffig = ''.join([UT.fig_dir(), 'catalogAbramson.SHMRsnap', str(nsnap), '.png'])
    fig.savefig(ffig, bbox_inches="tight") 
    plt.close()
    return None
    

def SHassignSFH(nsnap=15): 
    ''' Assign SFHs based on Msham at snapshot nsnap. Then investigate SHMR  
    at a later time/snapshot (lower redshift) 
    '''
    cat = Cat.CentralSubhalos(sigma_smhm=0.2, smf_source='li-march', nsnap0=15) 
    sh = cat.Read()
    
    i_la2016 = LA2016.SHassignSFH(sh, nsnap=nsnap, logMs_range=[9.,12.])
    hasmatch = (i_la2016 != -999) 

    # read in Louis's catalog
    cat = LA2016.catalogAbramson2016() 
    i_z = np.argmin(np.abs(cat['REDSHIFT'] - UT.z_nsnap(nsnap))) #  index that best match to snapshot 
    mstar = np.log10(cat['MSTEL_T'][:,i_z]) # M* @ z ~ zsnap
    i_zf = np.argmin(np.abs(cat['REDSHIFT'] - UT.z_nsnap(1))) #  index that best match to snapshot 
    mstarf = np.log10(cat['MSTEL_T'][:,i_zf]) # M* @ z ~ zsnap

    # now lets compare the resulting SHMRs 
    fig = plt.figure(figsize=(20,5))
    # first a scatter plot
    sub = fig.add_subplot(141) 
    sub.scatter(sh['halo.m.snap'+str(nsnap)][::10], sh['m.sham.snap'+str(nsnap)][::10], c='k', s=1, label='TreePM SHAM') 
    sub.scatter(sh['halo.m.snap'+str(nsnap)][hasmatch][::10], mstar[i_la2016[hasmatch]][::10], c='C1', s=1, label='assigned Abramson+(2016)') 
    sub.set_xlabel('log\,$M_h$', fontsize=25) 
    sub.set_xlim([11., 14.]) 
    sub.set_ylabel('log\,$M_*$', fontsize=25) 
    sub.set_ylim([9., 12.]) 
    sub.legend(loc=(0., 0.8), markerscale=5, handletextpad=0., frameon=True, fontsize=15) 

    # a DFM contour plot
    sub = fig.add_subplot(142)
    sub.scatter(sh['halo.m'][::10], sh['m.sham'][::10], c='k', s=1)
    sub.scatter(sh['halo.m'][hasmatch][::10], mstarf[i_la2016[hasmatch]][::10], c='C1', s=1)
    #DFM.hist2d(sh['halo.m.snap'+str(nsnap)], sh['m.sham.snap'+str(nsnap)], 
    #        levels=[0.68, 0.95], range=[[11., 15.], [9., 12.]], color='k', 
    #        bins=20, plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub) 
    #DFM.hist2d(sh['halo.m.snap'+str(nsnap)][hasmatch], mstar[i_la2016[hasmatch]],
    #        levels=[0.68, 0.95], range=[[11., 15.], [9., 12.]], color='C1', 
    #        bins=20, plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub) 
    sub.set_xlabel('log\,$M_h$', fontsize=25) 
    sub.set_xlim([11., 14.]) 
    sub.set_ylim([9., 12.]) 

    # SHMR plot
    smhmr = Obvs.Smhmr()
    sub = fig.add_subplot(143)
    mbin = np.arange(11., 15., 0.2) 
    ms_nonzero = (sh['m.sham.snap'+str(nsnap)] > 0.) 
    #m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(sh['halo.m.snap'+str(nsnap)][ms_nonzero], 
    #        sh['m.sham.snap'+str(nsnap)][ms_nonzero], m_bin=mbin)
    m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(sh['halo.m.snap'+str(nsnap)][hasmatch & ms_nonzero], 
            sh['m.sham.snap'+str(nsnap)][hasmatch & ms_nonzero], m_bin=mbin)
    sub.errorbar(m_mid, mu_logMs, sig_logMs, fmt='.k') 
    m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(sh['halo.m.snap'+str(nsnap)][hasmatch], 
            mstar[i_la2016[hasmatch]], m_bin=mbin)
    #m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(sh['halo.m.snap'+str(nsnap)][hasmatch], 
    #        sh['m.sham.snap'+str(nsnap)][hasmatch], m_bin=mbin)
    sub.errorbar(m_mid[m_mid > 11.8], mu_logMs[m_mid > 11.8], sig_logMs[m_mid > 11.8], fmt='.C1') 
    sub.set_xlabel('log $M_h$', fontsize=25) 
    sub.set_xlim([11., 14.]) 
    #sub.set_ylabel('log $M_*$', fontsize=25) 
    sub.set_ylim([9., 12.]) 
    #sub.set_ylim([0., 0.5]) 

    mh_slice = ((sh['halo.m.snap'+str(nsnap)] > 12.5) & (sh['halo.m.snap'+str(nsnap)] < 12.7)) 
    print sh['m.sham.snap'+str(nsnap)][mh_slice].min(), sh['m.sham.snap'+str(nsnap)][mh_slice].max()

    sub = fig.add_subplot(144)
    mbin = np.arange(11.8, 14., 0.1) 
    #m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(sh['halo.m.snap'+str(nsnap)][ms_nonzero], 
    #        sh['m.sham.snap'+str(nsnap)][ms_nonzero], m_bin=mbin)
    m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(sh['halo.m.snap'+str(nsnap)][hasmatch & ms_nonzero], 
            sh['m.sham.snap'+str(nsnap)][hasmatch & ms_nonzero], m_bin=mbin)
    sub.plot(m_mid, sig_logMs, c='k') 
    m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(sh['halo.m.snap'+str(nsnap)][hasmatch], 
            mstar[i_la2016[hasmatch]], m_bin=mbin)
    sub.plot(m_mid, sig_logMs, c='C1') 
    sub.set_xlabel('log $M_h$', fontsize=25) 
    sub.set_xlim([11., 14.]) 
    sub.set_ylabel('$\sigma_{\log\,M_*}$', fontsize=25) 
    sub.set_ylim([0., 0.5]) 

    ffig = ''.join([UT.fig_dir(), 'shcatalog.assignAbramson.SHMRsnap', str(nsnap), '.png'])
    fig.savefig(ffig, bbox_inches="tight") 
    plt.close()
    return None
    

if __name__=="__main__": 
    #LAbramCatalog()
    #matchSMF(n_boot=100)
    assignHalo(nsnap=15)
    #SHassignSFH(nsnap=15)
