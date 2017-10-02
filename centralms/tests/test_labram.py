import numpy as np
import time 

import env 
import util as UT
import observables as Obv
import catalog as Cats
import labram as LA

import matplotlib.pyplot as plt
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


def matchSMF(nsnap): 
    ''' ***TESTED***
    testing that the LA.matchSMF weights 
    '''
    mstar, _ = LA.read_LA(nsnap) # read in LA's galaxies at nsnap
    ws = LA.matchSMF(nsnap)
    
    fig = plt.figure(figsize=(8,8))
    sub = fig.add_subplot(111)
    
    prettyplot() 
    mbin = np.arange(9.0, 12.1, 0.1)
    wgal, bbb = np.histogram(mstar, weights=ws, bins=mbin)
    phi = Obv.analyticSMF(UT.z_nsnap(nsnap), m_arr=mbin, dlogm=0.1, source='li-march')
    sub.scatter(phi[0], phi[1], lw=0, c='k', s=40)
    sub.plot(0.5*(bbb[:-1]+bbb[1:]), wgal, ls='--', c='r')
    sub.set_xlim([9., 12.])
    sub.set_xlabel('$\mathtt{log\;M_*}$', fontsize=25)
    sub.set_ylim([1e-5, 3e-2])
    sub.set_ylabel('$\Phi$', fontsize=25)
    sub.set_yscale("log")
    plt.show() 
    plt.close() 
    return None


def assignHalo(method='random', SHAM_sig=0.2, nsnap0=15): 
    ''' 
    '''
    t0 = time.time() 
    cat = Cats.SubhaloHistory(sigma_smhm=SHAM_sig, nsnap_ancestor=nsnap0)
    halos = cat.Read()
    print time.time() - t0
    mstar, _ = LA.read_LA(nsnap0)
    ws = LA.matchSMF(1) # weights to match snapshot 1 SMFs

    i_assign = LA.assignHalo(nsnap0, halos, method=method)
    matched = np.where(i_assign != -999)
    
    mhalo = np.array([halos['snapshot'+str(nsnap0)+'_m.max'][i] for i in i_assign[matched]])
    mstar = mstar[matched]
    ws = ws[matched]
    
    prettyplot()
    fig = plt.figure(figsize=(15,15))
    # plot M* - Mhalo at z ~ 1
    sub = fig.add_subplot(221)
    sub.scatter(mhalo, mstar) 
    sub.set_xlim([11., 15.])
    sub.set_xlabel('halo mass', fontsize=25)
    sub.set_ylim([9., 12.])
    sub.set_ylabel('stellar mass', fontsize=25)
    
    # plot sigma_log M* at z ~ 1
    sub = fig.add_subplot(222)
    mh_lows = np.arange(11., 15., 0.5)
    mu_mstar = np.zeros(len(mh_lows))
    var_mstar = np.zeros(len(mh_lows))
    for im, mh_low in enumerate(mh_lows): 
        in_mhbin = np.where((mhalo >= mh_low) & (mhalo < mh_low+0.5))
        if np.sum(ws[in_mhbin]) > 0.: 
            mu_mstar[im] = np.sum(mstar[in_mhbin] * ws[in_mhbin]) / np.sum(ws[in_mhbin])
            var_mstar[im] = np.average((mstar[in_mhbin] - mu_mstar[im])**2, weights=ws[in_mhbin])
    
    sub.plot(mh_lows+0.25, np.sqrt(var_mstar))
    sub.set_xlim([11., 15.])
    sub.set_xlabel('halo mass', fontsize=25)
    sub.set_ylim([0., 0.5])
    sub.set_ylabel('$\sigma_\mathtt{log\;M_*}$', fontsize=25)

    mhalo_f = np.array([halos['m.max'][i] for i in i_assign[matched]])
    mstar_f, _ = LA.read_LA(1)
    mstar_f = mstar_f[matched]

    # plot M* - Mhalo at z ~ 0
    sub = fig.add_subplot(223)
    sub.scatter(mhalo_f, mstar_f) 
    sub.set_xlim([11., 15.])
    sub.set_xlabel('halo mass', fontsize=25)
    sub.set_ylim([9., 12.])
    sub.set_ylabel('stellar mass', fontsize=25)

    # plot sigma_log M* at z ~ 0
    sub = fig.add_subplot(224)
    mu_mstar_f = np.zeros(len(mh_lows))
    var_mstar_f = np.zeros(len(mh_lows))
    for im, mh_low in enumerate(mh_lows): 
        in_mhbin = np.where((mhalo_f >= mh_low) & (mhalo_f < mh_low+0.5))
        if np.sum(ws[in_mhbin]) > 0.: 
            mu_mstar_f[im] = np.sum(mstar_f[in_mhbin] * ws[in_mhbin]) / np.sum(ws[in_mhbin])
            var_mstar_f[im] = np.average((mstar_f[in_mhbin] - mu_mstar[im])**2, weights=ws[in_mhbin])
    
    sub.plot(mh_lows+0.25, np.sqrt(var_mstar_f))
    sub.set_xlim([11., 15.])
    sub.set_xlabel('halo mass', fontsize=25)
    sub.set_ylim([0., 0.5])
    sub.set_ylabel('$\sigma_\mathtt{log\;M_*}$', fontsize=25)

    f_fig = ''.join([UT.fig_dir(), 'LAbram.assignHalo.method_', method, '.snap', str(nsnap0), '.png'])
    fig.savefig(f_fig, bbox_inches='tight')
    plt.close() 
    return None


if __name__=="__main__": 
    assignHalo(method='dmhalo')
