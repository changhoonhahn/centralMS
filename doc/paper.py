#!/usr/bin/env python
''' 

figures for centralMS paper 

'''
import os
import h5py
import pickle
import numpy as np 
import corner as DFM
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal as MNorm
from letstalkaboutquench.fstarforms import fstarforms
# -- centralms -- 
from centralms import util as UT
from centralms import sfh as SFH 
from centralms import abcee as ABC
from centralms import catalog as Cat
from centralms import evolver as Evol
from centralms import labram as LA2016
from centralms import observables as Obvs
# -- matplotlib -- 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from ChangTools.plotting import prettycolors
from matplotlib.patches import Rectangle
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
mpl.rcParams['hatch.linewidth'] = 0.3  


def groupcatSFMS(mrange=[10.6,10.8]): 
    '''Figure of the z~0 group catalog. 
    Panel a) SFR-M* relation 
    Panel b) P(SSFR) with SFMS fitting 
    '''
    # Read in Jeremy's group catalog  with Mr_cut = -18
    gc = Cat.Observations('group_catalog', Mrcut=18, position='central')
    gc_cat = gc.Read() 
    fig = plt.figure(figsize=(10,5)) 

    # fit the SFMS using lettalkaboutquench sfms fitting
    _fSFMS = fstarforms() 
    _fit_logm, _fit_logsfr = _fSFMS.fit(gc_cat['mass'], gc_cat['sfr'], method='gaussmix', fit_range=None)
    logsfr_ms = _fSFMS.powerlaw(logMfid=10.5) 
    print _fSFMS._powerlaw_m
    print _fSFMS._powerlaw_c

    fSFMS = fstarforms() 
    fit_logm, _ = fSFMS.fit(gc_cat['mass'], gc_cat['sfr'], method='gaussmix', fit_range=mrange)
    _, fit_fsfms = fSFMS.frac_SFMS()
    i_fit = np.abs(fit_logm - np.mean(mrange)).argmin()

    # log SFR - log M* highlighting where the SFMS lies 
    sub1 = fig.add_subplot(1,2,1)
    DFM.hist2d(gc_cat['mass'], gc_cat['sfr'], color='#ee6a50',
            levels=[0.68, 0.95], range=[[9., 12.], [-3.5, 1.5]], 
            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub1) 
    gc = Cat.Observations('group_catalog', Mrcut=18, position='central')
    gc_cat = gc.Read() 
    #sub1.vlines(mrange[0], -5., 2., color='k', linewidth=2, linestyle='--')
    #sub1.vlines(mrange[1], -5., 2., color='k', linewidth=2, linestyle='--')
    #sub1.fill_between(mrange, [2.,2.], [-5.,-5], color='#1F77B4', alpha=0.25)
    sub1.fill_between(mrange, [2.,2.], [-5.,-5], color='k', linewidth=0, alpha=0.25)
    print _fit_logm, _fit_logsfr
    sub1.plot(np.linspace(9.8, 11., 10), logsfr_ms(np.linspace(9.8, 11., 10)), c='k', linestyle='--') 
    sub1.set_xticks([9., 10., 11., 12.])
    sub1.set_xlabel('log$(\; M_*\; [M_\odot]\;)$', fontsize=20)
    sub1.set_yticks([-3., -2., -1., 0., 1.])
    sub1.set_ylabel('log$(\; \mathrm{SFR}\; [M_\odot/\mathrm{yr}]\;)$', fontsize=20)
    sub1.text(0.95, 0.1, 'SDSS central galaxies',
            ha='right', va='center', transform=sub1.transAxes, fontsize=20)

    # P(log SSFR) 
    sub2 = fig.add_subplot(1,2,2)
    inmbin = np.where((gc_cat['mass'] > mrange[0]) & (gc_cat['mass'] < mrange[1]))
    bedge, pp = np.histogram(gc_cat['ssfr'][inmbin], range=[-14., -9.], bins=32, normed=True)
    pssfr = UT.bar_plot(pp, bedge)
    sub2.plot(pssfr[0], pssfr[1], c='k', lw=2) 
    # overplot GMM component for SFMS
    gmm_weights = fSFMS._gmix_weights[i_fit]
    gmm_means = fSFMS._gmix_means[i_fit]
    gmm_vars = fSFMS._gmix_covariances[i_fit]
    icomp = gmm_means.argmax()
    xx = np.linspace(-14., -9, 100)
    sub2.fill_between(xx, np.zeros(len(xx)), gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp]), 
            color='#1F77B4', linewidth=0)

    for i_comp in range(len(gmm_vars)): 
        if i_comp == 0: 
            gmm_tot = gmm_weights[i_comp]*MNorm.pdf(xx, gmm_means[i_comp], gmm_vars[i_comp])
        else: 
            gmm_tot += gmm_weights[i_comp]*MNorm.pdf(xx, gmm_means[i_comp], gmm_vars[i_comp])
    
    #sub2.plot(xx, gmm_tot, color='r', linewidth=2)

    sub2.set_xlim([-13.25, -9.5]) 
    sub2.set_xticks([-10., -11., -12., -13.][::-1])
    #sub2.set_xlim([-9.5, -13.25]) 
    #sub2.set_xticks([-10., -11., -12., -13.])
    sub2.set_xlabel('log$(\; \mathrm{SSFR}\; [\mathrm{yr}^{-1}]\;)$', fontsize=20)
    sub2.set_ylim([0., 1.5]) 
    sub2.set_yticks([0., 0.5, 1., 1.5])
    sub2.set_ylabel('$p\,(\;\mathrm{log}\; \mathrm{SSFR}\;)$', fontsize=20)
    # mass bin 
    sub2.text(0.5, 0.9, '$'+str(mrange[0])+'< \mathrm{log}\, M_* <'+str(mrange[1])+'$',
            ha='center', va='center', transform=sub2.transAxes, fontsize=20)
    sub2.text(0.9, 0.33, '$f_\mathrm{SFMS}='+str(round(fit_fsfms[i_fit],2))+'$',
            ha='right', va='center', transform=sub2.transAxes, fontsize=20)
    fig.subplots_adjust(wspace=.3)
    fig.savefig(''.join([UT.tex_dir(), 'figs/groupcat.pdf']), bbox_inches='tight', dpi=150) 
    plt.close()
    return None


def fQ_fSFMS(logMfid=10.5): 
    ''' Figure comparing the quiescent fraction based on "traditional" SFMS cut 
    to the SFMS fraction. 
    '''
    # Read in Jeremy's group catalog  with Mr_cut = -18
    gc = Cat.Observations('group_catalog', Mrcut=18, position='central')
    gc_cat = gc.Read() 

    # fit the SFMS using lettalkaboutquench sfms fitting
    fSFMS = fstarforms() 
    fit_logm, fit_logsfr = fSFMS.fit(gc_cat['mass'], gc_cat['sfr'], method='gaussmix')
    _, fit_fsfms = fSFMS.frac_SFMS()

    # now fit a fSFMS(M*) 
    coeff = np.polyfit(fit_logm-logMfid, fit_fsfms, 1)

    # output f_SFMS to data (for posterity)
    f = open(''.join([UT.tex_dir(), 'dat/fsfms.dat']), 'w') 
    f.write('### header ### \n') 
    f.write('star-formation main sequence (SFMS) fraction: fraction of galaxies \n') 
    f.write('within a log-normal fit of the SFMS. See paper for details.\n') 
    f.write('best-fit f_SFMS = '+str(round(coeff[0], 3))+' (log M* - '+str(logMfid)+') + '+str(round(coeff[1],3))+'\n')
    f.write('columns: log M*, f_SFMS\n') 
    f.write('### header ### \n') 
    for i_m in range(len(fit_logm)): 
        f.write('%f \t %f' % (fit_logm[i_m], fit_fsfms[i_m]))
        f.write('\n')
    f.close() 
    
    # best-fit quiescent fraction from Hahn et al. (2017) 
    f_Q_cen = lambda mm: -6.04 + 0.64 * mm 
    
    pretty_colors = prettycolors()  
    fig = plt.figure(figsize=(5,5)) 
    sub = fig.add_subplot(111)
    sub.scatter(fit_logm, 1. - fit_fsfms, lw=0, c=pretty_colors[1]) 
    marr = np.linspace(9., 12., 100) 
    fsfms_bf = sub.plot(marr, 1-(coeff[0]*(marr-logMfid) + coeff[1]), c=pretty_colors[1], lw=2, ls='-')
    fq = sub.plot(marr, f_Q_cen(marr), c=pretty_colors[3], lw=1.5, ls='--')
    sub.set_xlim([9., 11.5])
    sub.set_xticks([9., 10., 11.])
    sub.set_xlabel('log$(\; M_*\; [M_\odot]\;)$', fontsize=25)
    sub.set_ylim([0., 1.])
    sub.set_ylabel('$1 - f_\mathrm{SFMS}$', fontsize=25)

    legend1 = sub.legend(fsfms_bf, ['$1 - f^\mathrm{bestfit}_\mathrm{SFMS}$'], loc='upper left', prop={'size': 20})
    sub.legend(fq, ['Hahn et al.(2017)\n $f_\mathrm{Q}^\mathrm{cen}(M_*, z\sim0)$'], loc='lower right', prop={'size': 15})
    plt.gca().add_artist(legend1)

    fig.savefig(''.join([UT.tex_dir(), 'figs/fq_fsfms.pdf']), bbox_inches='tight', dpi=150) 
    plt.close()
    return None


def SFHmodel(nsnap0=15, downsampled='20'):
    ''' Figure that illustrates the SFH of galaxies. 
    Two panel plot. Panel a) SFH of a galaxy plotted alongside SFMS 
    '''
    fig = plt.figure(figsize=(10,4.5))
    # log SFR - log M* galaxy evolution 
    sub1 = fig.add_subplot(122)
    # Delta log SFR(t) evolution 
    sub2 = fig.add_subplot(121)
    sub2.fill_between([5., 14.], [0.3, 0.3], [-0.3, -0.3], color='k', alpha=0.15, linewidth=0)
    
    for i_m, method in enumerate(['randomSFH1gyr.sfsflex', 'randomSFH5gyr.sfsflex']): 
        # median theta 
        if i_m == 0: 
            abcout = ABC.readABC(method, 14)
            theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) for i in range(len(abcout['theta'][0]))]
        tt = ABC._model_theta(method, theta_med)

        subcat, tsteps, dlogSFR_amp = ABC.model(method, theta_med, nsnap0=nsnap0, downsampled=downsampled, testing=True) 

        # randomly pick a galaxy that match the below criteria
        isSF = np.where(subcat['galtype'] == 'sf')[0]
        elig = ((subcat['nsnap_start'][isSF] == nsnap0) & 
                (subcat['weights'][isSF] > 0.) & 
                (np.amax(np.abs(dlogSFR_amp), axis=1) < 0.3) & 
                (subcat['m.star'][isSF] > 10.5+0.2*float(i_m)) & 
                (subcat['m.star'][isSF] < 10.6+0.2*float(i_m)))
        ielig = np.random.choice(np.arange(len(isSF))[elig], size=1)[0]
        i_gal = isSF[ielig]
        
        # plot dlogSFR_MS
        if i_m == 0: 
            sub2.plot([UT.t_nsnap(nsnap0), UT.t_nsnap(1)], [0.,0.], ls='--', c='k')
        if '1gyr' in method:
            lbl = '$t_\mathrm{duty} = 1\,\mathrm{Gyr}$'
        elif '5gyr' in method: 
            lbl = '$t_\mathrm{duty} = 5\,\mathrm{Gyr}$'
        xx, yy = [], []
        for i in range(dlogSFR_amp.shape[1]-1):
            xx.append(tsteps[ielig,i]) 
            yy.append(dlogSFR_amp[ielig,i])
            xx.append(tsteps[ielig,i+1]) 
            yy.append(dlogSFR_amp[ielig,i])
        sub2.plot(xx, yy, c='C'+str(i_m), label=lbl)

        # track back it's M* and SFR
        mstar_hist, sfr_hist = [subcat['m.star0'][i_gal]], [subcat['sfr0'][i_gal]] 
        for isnap in range(2,nsnap0)[::-1]: 
            mstar_hist.append(subcat['m.star.snap'+str(isnap)][i_gal])
            sfr_hist.append(subcat['sfr.snap'+str(isnap)][i_gal])
        mstar_hist.append(subcat['m.star'][i_gal]) 
        sfr_hist.append(subcat['sfr'][i_gal]) 
        sfr_hist = np.array(sfr_hist)
        
        # SFMS 
        sfr_sfms = [SFH.SFR_sfms(mstar_hist[0], UT.z_nsnap(nsnap0), tt['sfms'])]
        for ii, isnap in enumerate(range(2,nsnap0)[::-1]): 
            sfr_sfms.append(SFH.SFR_sfms(mstar_hist[ii+1], UT.z_nsnap(isnap), tt['sfms']))
        sfr_sfms.append(SFH.SFR_sfms(mstar_hist[-1], UT.z_nsnap(1), tt['sfms']))
        sfr_sfms = np.array(sfr_sfms)
        f_sfms = interp1d(mstar_hist, sfr_sfms, kind='slinear') #smooth
        t_mstar = interp1d(mstar_hist, UT.t_nsnap(range(1,nsnap0+1)[::-1]))

        marr = np.linspace(mstar_hist[0], mstar_hist[-1], 200)
        sub1.plot(10**marr, f_sfms(marr), c='k', lw=1, ls='--', label='$\log\,\overline{\mathrm{SFR}}_\mathrm{SFS}$')
        if i_m == 0: 
            sub1.legend(loc='lower right', handletextpad=0.1, fontsize=15)
        def dlogSFR_t(ttt):
            t_steps = tsteps[ielig,:]
            dlogSFRamp = dlogSFR_amp[ielig,:]
            ishift = np.abs(t_steps - ttt).argmin()
            closest = t_steps[ishift]
            if closest > ttt: 
                ishift -= 1 
            return dlogSFRamp[ishift]

        sub1.plot(10**marr, f_sfms(marr)+np.array([dlogSFR_t(ttt) for ttt in t_mstar(marr)]), c='C'+str(i_m))
        # plot SFMS(M*, z)
        m_arr = np.linspace(9., 12., 100)
        zs = np.arange(0., 1.2, 0.25)
        zs[0] = 0.05
        if i_m == 0: 
            for z in zs: 
                mslope = SFH.SFR_sfms(11., z, tt['sfms']) - SFH.SFR_sfms(10., z, tt['sfms'])
                sub1.plot(10**m_arr, SFH.SFR_sfms(m_arr, z, tt['sfms']),
                        c='k', ls=':', lw=0.75)
                sub1.text(10**10.05, SFH.SFR_sfms(10.05, z, tt['sfms']), '$z = '+str(z)+'$', 
                        rotation=0.5*np.arctan(mslope)*180./np.pi, fontsize=12, va='bottom')

    sub1.set_xlabel('$\; M_*\; [M_\odot]$', fontsize=25)
    sub1.set_xscale('log') 
    sub1.set_xlim([10**10, 10**11])
    sub1.set_xticks([10**10, 10**11])
    sub1.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    sub1.get_xaxis().get_major_formatter().labelOnlyBase = False
    #sub1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    sub1.set_ylim([-0.4, 1.6])
    sub1.set_yticks([-0.4, 0., 0.4, 0.8, 1.2, 1.6]) 
    sub1.set_ylabel('log $(\; \mathrm{SFR}\; [M_\odot/\mathrm{yr}]\;)$', fontsize=25)
    sub1.yaxis.tick_right()
    sub1.yaxis.set_label_position("right")

    sub2.set_xlim([UT.t_nsnap(nsnap0), UT.t_nsnap(1)]) 
    sub2.set_xlabel('$t_\mathrm{cosmic}\;[\mathrm{Gyr}]$', fontsize=25)
    sub2.set_ylim([-1., 1.]) 
    sub2.set_yticks([-1., -0.5, 0., 0.5, 1.])
    sub2.set_ylabel('$\Delta$ log $(\;\mathrm{SFR}\;[M_\odot/\mathrm{yr}]\;)$', fontsize=25)
    sub2.legend(loc='lower right', prop={'size':15})
    fig.subplots_adjust(wspace=0.1)
    fig.savefig(''.join([UT.tex_dir(), 'figs/sfh_pedagogical.pdf']), bbox_inches='tight', dpi=150) 
    return None 


def Illustris_SFH(): 
    ''' Figure that uses Illustris SFHs to justify our model of galaxies 
    '''
    # read in illustris SFH file from Tijske
    dat = h5py.File(UT.dat_dir()+'binsv2all1e8Msunh_z0.hdf5', 'r')

    # formed stellar mass is in a grid of time bins and metallicity
    t_bins = np.array([0.0, 0.005, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.125,0.175,0.225,0.275,0.325,0.375,0.425,0.475,0.55,0.65,0.75,0.85,0.95,1.125,1.375,1.625,1.875,2.125,2.375,2.625,2.875,3.125,3.375,3.625,3.875,4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.25,9.75,10.25,10.75,11.25,11.75,12.25,12.75,13.25,13.75])
    
    galpop = {}
    galpop['M*'] = dat['CurrentStellarMass'].value.flatten() * 1e10 # current stellar mass
    # calculate SFRs
    # first sum up all the metallicities so you have delta M* in a grid of galaxies and time 
    # then average SFR over the 0.015 Gyr time period 
    sfh_grid = dat['FormedStellarMass'].value
    dM_t = np.sum(sfh_grid, axis=1) 

    #galpop['sfr'] = (1e10 * (dM_t[:,0] + dM_t[:,1])/(0.015 * 1e9)).flatten() 
    sfhs = np.zeros((len(galpop['M*']), len(t_bins)-2))
    t_mid = np.zeros(len(t_bins)-2)
    # z=0 SFR averaged over 150Myr
    sfhs[:,0] = (1e10 * (dM_t[:,0] + dM_t[:,1])/(0.015 * 1e9)).flatten() 
    sfhs[:,1:] = 10.*(dM_t[:,2:-1]/(t_bins[3:] - t_bins[2:-1]))
    t_mid[0] = 0.0075
    t_mid[1:] = 0.5 *(t_bins[3:] + t_bins[2:-1])

    # stellar mass one timestep ago 
    M_t_1 = np.zeros((len(galpop['M*']), len(t_bins)-1))
    M_t_1[:,0] = galpop['M*'] - 1e10 * (dM_t[:,0] + dM_t[:,1]) 
    for i in range(1,M_t_1.shape[1]): 
        M_t_1[:,i] = M_t_1[:,i-1] - 1e10 * dM_t[:,i+1]
    # stellar mass history at t_mid 
    Msh = 0.5 * (M_t_1[:,:-1] + M_t_1[:,1:]) 
    
    # calculate delta log SFR for galaxies by fitting SFMS every 10 time bins 
    dlogsfrs, t_skip = [], [] 
    for i in range(len(t_mid)-12): 
        fSFMS = fstarforms() 
        sfr_lim = (sfhs[:,i] > 0.) 
        fit_logm_i, fit_logsfr_i, _ = fSFMS.fit(
                np.log10(Msh[sfr_lim,i]), 
                np.log10(sfhs[sfr_lim,i]),
                method='gaussmix', 
                fit_range=[9.0, 10.5], 
                fit_error='bootstrap',
                n_bootstrap=10, 
                dlogm=0.2)
        sfms_fit_i = fSFMS.powerlaw(logMfid=10.5)

        fig = plt.figure() 
        sub = fig.add_subplot(111)
        DFM.hist2d(np.log10(Msh[:,i]), np.log10(sfhs[:,i]),
                levels=[0.68, 0.95], range=[[9., 12.], [-3., 2.]], 
                bins=16, ax=sub) 
        sub.scatter(fit_logm_i, fit_logsfr_i, c='r', marker='x')
        sub.text(0.95, 0.1, '$t_\mathrm{cosmic} = '+str(13.75 - t_mid[i])+'$',
                ha='right', va='center', transform=sub.transAxes, fontsize=20)

        m_arr = np.linspace(9.0, 12.0, 20) 
        sub.plot(m_arr, sfms_fit_i(m_arr), c='k', lw=2, ls='--') 
        fig.savefig(''.join([UT.fig_dir(), 'illustris.sfms.', str(i), '.png']), 
                bbox_inches='tight') 
        plt.close() 

        if fSFMS._powerlaw_m < 0.5 or fSFMS._powerlaw_m > 1.5 or len(fit_logm_i) < 3: 
            continue

        dlogsfrs.append(np.log10(sfhs[:,i]) - sfms_fit_i(np.log10(Msh[:,i]))) 
        t_skip.append(t_mid[i]) 

    # now fit SFMS at z ~ 0  
    fSFMS = fstarforms() 
    sfr_lim = (sfhs[:,0] > 0.) 
    fit_logm, fit_logsfr, _ = fSFMS.fit(np.log10(galpop['M*'][sfr_lim]), np.log10(sfhs[sfr_lim,0]), 
            method='gaussmix', fit_range=[9.0, 10.5], dlogm=0.2, 
            fit_error='bootstrap', n_bootstrap=10)
    sfms_fit = fSFMS.powerlaw(logMfid=10.5)

    # star-forming galaxies at z ~ 0 
    z0sf = ((np.log10(sfhs[:,0]) > sfms_fit(np.log10(galpop['M*']))-0.6) & 
            (np.log10(galpop['M*']) > 10.5) &  (np.log10(galpop['M*']) < 10.6)) 
    #z0q = ((np.log10(sfhs[:,0]) < sfms_fit(np.log10(galpop['M*']))-0.9) & 
    #        (np.log10(galpop['M*']) > 10.5) &  (np.log10(galpop['M*']) < 10.6)) 

    fig = plt.figure()
    sub = fig.add_subplot(111)
    iarr = [0, 12, 17, 19]+range(22,42)
    for i in np.arange(len(z0sf))[z0sf]: 
        sub.plot((t_bins[-1] - t_skip)[iarr], np.array(dlogsfrs)[:,i][iarr], c='k', alpha=0.1, lw=0.1) 
    z0sf = ((np.log10(sfhs[:,0]) > sfms_fit(np.log10(galpop['M*']))-0.6) & 
            (np.log10(galpop['M*']) > 10.5) &  (np.log10(galpop['M*']) < 10.6)) 
    for ii, i in enumerate(np.random.choice(np.arange(len(z0sf))[z0sf], 10)): 
        sub.plot((t_bins[-1] - t_skip)[iarr], np.array(dlogsfrs)[:,i][iarr], lw=1, c='C'+str(ii)) 
    sub.set_xlim([11., 13.75]) 
    sub.set_xlabel('$t_\mathrm{cosmic}$ [Gyr]', fontsize=25) 
    sub.set_ylim([-.6, .6]) 
    sub.set_yticks([-0.4, 0., 0.4]) 
    sub.set_ylabel('$\Delta$ log $(\;\mathrm{SFR}\;[M_\odot/\mathrm{yr}]\;)$', fontsize=25)
    fig.savefig(''.join([UT.tex_dir(), 'figs/illustris_sfh.pdf']), bbox_inches='tight', dpi=150) 
    plt.close() 
    return None 


def SFMSprior_z1():
    ''' Compare SFMS from literature at z~1 with the prior of 
    the SFMS in our ABC
    '''
    prior_min = [1., 0.4]
    prior_max = [2., 0.8]

    # Lee et al. (2015)
    logSFR_lee = lambda mm: 1.53 - np.log10(1 + (10**mm/10**(10.10))**-1.26)
    # Noeske et al. (2007) 0.85 < z< 1.10 (by eye)
    logSFR_noeske = lambda mm: (1.580 - 1.064)/(11.229 - 10.029)*(mm - 10.0285) + 1.0637
    # Moustakas et al. (2013) 0.8 < z < 1. (by eye)  
    logSFR_primus = lambda mm: (1.3320 - 1.296)/(10.49 - 9.555) * (mm-9.555) + 1.297
    # Hahn et al. (2017)  
    logSFR_hahn = lambda mm: 0.53*(mm-10.5) + 1.1*(1.-0.05) - 0.11
    
    def logSFR_prior_min(mm): 
        sfr1 = SFH.SFR_sfms(mm, 1., {'zslope': prior_min[0], 'mslope': prior_min[1]})
        sfr2 = SFH.SFR_sfms(mm, 1., {'zslope': prior_min[0], 'mslope': prior_max[1]})
        sfr3 = SFH.SFR_sfms(mm, 1., {'zslope': prior_max[0], 'mslope': prior_min[1]})
        sfr4 = SFH.SFR_sfms(mm, 1., {'zslope': prior_max[0], 'mslope': prior_max[1]})
        return np.min([sfr1, sfr2, sfr3, sfr4]) 
    
    def logSFR_prior_max(mm): 
        sfr1 = SFH.SFR_sfms(mm, 1., {'zslope': prior_min[0], 'mslope': prior_min[1]})
        sfr2 = SFH.SFR_sfms(mm, 1., {'zslope': prior_min[0], 'mslope': prior_max[1]})
        sfr3 = SFH.SFR_sfms(mm, 1., {'zslope': prior_max[0], 'mslope': prior_min[1]})
        sfr4 = SFH.SFR_sfms(mm, 1., {'zslope': prior_max[0], 'mslope': prior_max[1]})
        return np.max([sfr1, sfr2, sfr3, sfr4]) 
    fig = plt.figure()
    sub = fig.add_subplot(111)
    marr = np.linspace(9., 12., 100)
    sub.plot(marr, logSFR_noeske(marr), label='Noeske et al.(2007)')
    sub.plot(marr, logSFR_primus(marr), label='Moustakas et al.(2013)')
    sub.plot(marr, logSFR_lee(marr), label='Lee et al.(2015)')
    sub.plot(marr, logSFR_hahn(marr), label='Hahn et al.(2017)')
    sub.fill_between(marr, 
            [logSFR_prior_min(m) for m in marr], 
            [logSFR_prior_max(m) for m in marr], 
            label='Prior', alpha=0.5)
    sub.legend(loc='lower right') 
    sub.set_xlim([9.75, 11.5])
    sub.set_xticks([10., 10.5, 11., 11.5])
    sub.set_xlabel('log$(\; M_*\; [M_\odot]\;)$', fontsize=20)
    sub.set_ylim([0., 3.])
    sub.set_ylabel('log $(\;\mathrm{SFR}\;[M_\odot/\mathrm{yr}])$', fontsize=20)
    fig.savefig(''.join([UT.tex_dir(), 'figs/SFMSprior.z1.pdf']), 
            bbox_inches='tight', dpi=150) 
    plt.close() 
    return None


def qaplotABC(runs=None, Ts=None, dMhalo=0.1): 
    ''' Figure that illustrates how the ABC fitting works using two different
    runs overplotted on it each other
    '''
    nsnap0 = 15
    sigma_smhm = 0.2
    sumstat = ['sfsmf']

    # summary statistics of data (i.e. the SMF of SF centrals) 
    marr, smf, phi_err = Obvs.dataSMF(source='li-white') # Li & White SMF 
    fcen = (1. - np.array([Obvs.f_sat(mm, 0.05) for mm in marr])) # sallite fraction 
    fsf = np.clip(Evol.Fsfms(marr), 0., 1.) 
    phi_sf_cen = fsf * fcen * smf 
    phi_err *= np.sqrt(1./(1.-np.array([Obvs.f_sat(mm, 0.05) for mm in marr]))) # now scale err by f_cen 
    fsfs = np.clip(Evol.Fsfms(marr), 0., 1.) 
    fsfs_errscale = np.ones(len(marr))
    fsfs_errscale[fsfs < 1.] = np.sqrt(1./(1.-fsfs[fsfs < 1.]))
    phi_err *= fsfs_errscale
        
    # read in model(theta_median)
    sims_meds = []   
    for run, T in zip(runs, Ts): 
        sim_meds = [] 
        for i in range(100): 
            f = pickle.load(open(''.join([UT.dat_dir()+'abc/'+run+'/model/', 
                'model.theta', str(i), '.t', str(T), '.p']), 'rb'))
            sim_i = {} 
            for key in f.keys(): 
                sim_i[key] = f[key]
            sim_meds.append(sim_i) 
        sims_meds.append(sim_meds)

    colors = ['C3', 'C0']
    #labels = [r'model($\theta_\mathrm{median}$)', r'model($\theta_\mathrm{median}$)']
    labels = [None for r in runs]
    if 'nodutycycle.sfsmf.sfsbroken' in runs: 
        labels[runs.index('nodutycycle.sfsmf.sfsbroken')] = 'No duty cycle'
    if 'randomSFH1gyr.sfsmf.sfsbroken' in runs: 
        labels[runs.index('randomSFH1gyr.sfsmf.sfsbroken')] = '$t_\mathrm{duty} = 1\,$Gyr'
    if 'randomSFH10gyr.sfsmf.sfsbroken' in runs: 
        labels[runs.index('randomSFH10gyr.sfsmf.sfsbroken')] = '$t_\mathrm{duty} = 10\,$Gyr'

    fig = plt.figure(figsize=(16,5))
    # --- SMF panel ---
    sub = fig.add_subplot(131)
    sub.fill_between(marr, phi_sf_cen-phi_err, phi_sf_cen+phi_err,
            linewidth=0., color='k', alpha=0.3, label=r'SDSS')#'$\Phi_\mathrm{SF, cen}^{\footnotesize \mathrm{SDSS}}$') 
    for i_s, meds in enumerate(sims_meds): 
        phi_sim = [] 
        for med in meds: 
            phi_sim_i = ABC.modelSum(med, sumstat=['sfsmf']) 
            phi_sim.append(phi_sim_i[0]) 
        sub.plot(marr, np.percentile(phi_sim, [50], axis=0)[0], c=colors[i_s]) #label=labels[i_s])#, label=r'model($\theta_\mathrm{median}$)')
    sub.plot([0.,0.], [0.,0.], c='k', label=r'model($\theta_\mathrm{median}$)')
    sub.set_xlim([9.4, 11.25])
    sub.set_xlabel('log $(\; M_*\; [M_\odot]\;)$', fontsize=25)
    sub.set_ylim([1e-5, 10**-2.])
    sub.set_yscale('log')
    sub.set_ylabel('log $(\;\Phi\; / \mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}\;)$', fontsize=25)
    sub.legend(loc='lower left', handletextpad=0.2, prop={'size':20}) 
    sub.text(0.95, 0.9, r'SF centrals', 
            ha='right', va='center', transform=sub.transAxes, fontsize=20)

    # --- SFMS panel ---
    sub = fig.add_subplot(132)
    for i_s, meds in enumerate(sims_meds): 
        med = meds[0] 
        isSF = np.where(med['galtype'] == 'sf') # only SF galaxies 
        DFM.hist2d(
                med['m.star'][isSF], 
                med['sfr'][isSF], 
                weights=med['weights'][isSF], 
                levels=[0.68, 0.95], range=[[9., 12.], [-2., 1.]], color=colors[i_s], 
                bins=10, plot_datapoints=False, fill_contours=False, plot_density=True, ax=sub) 
    sub.set_xlim([9., 11.])
    sub.set_xticks([9., 9.5, 10., 10.5, 11.]) 
    sub.set_xlabel('log $(\; M_*\; [M_\odot]\;)$', fontsize=25)
    sub.set_ylim([-2., 1.])
    sub.set_yticks([-2., -1., 0., 1.])
    sub.set_ylabel('log $(\;\mathrm{SFR}\;[M_\odot/\mathrm{yr}])$', fontsize=25)

    # --- sigma_logM* panel ---
    sub = fig.add_subplot(133)
    smhmr = Obvs.Smhmr()
    # simulation 
    mhalo_bin = np.linspace(10., 15., int(5./dMhalo)+1)
    for i_s, meds in enumerate(sims_meds): 
        for ii, med in enumerate(meds): 
            isSF = np.where(med['galtype'] == 'sf') # only SF galaxies 
            m_mid_i, _, sig_mhalo_i, cnt_i = smhmr.Calculate(med['halo.m'][isSF], med['m.star'][isSF], 
                    weights=med['weights'][isSF], m_bin=mhalo_bin)

            if ii == 0: 
                sig_mhalos = np.zeros((len(meds), len(cnt_i)))
                counts = np.zeros((len(meds), len(cnt_i)))
            sig_mhalos[ii,:] = sig_mhalo_i
            counts[ii,:] = cnt_i
        sig_mhalo_low = np.zeros(len(m_mid_i))
        sig_mhalo_high = np.zeros(len(m_mid_i))
        for im in range(len(m_mid_i)): 
            if np.mean(counts[:,im]) > 50.: 
                above_zero = np.where(counts[:,im] > 0) 
                sig_mhalo_low[im], sig_mhalo_high[im] = np.percentile((sig_mhalos[:,im])[above_zero], [16, 84])#, axis=0)
        above_zero = np.where(sig_mhalo_high > 0) 
        sub.fill_between(m_mid_i[above_zero], sig_mhalo_low[above_zero], sig_mhalo_high[above_zero], 
                color=colors[i_s], linewidth=0, alpha=0.5, label=labels[i_s]) 
    sub.set_xlim([11.5, 13])
    sub.set_xlabel('log $(\; M_{h}\; [M_\odot]\;)$', fontsize=25)
    sub.set_ylim([0.1, 0.5])
    sub.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5]) 
    sub.set_ylabel('$\sigma_{\mathrm{log}\,M_*}$', fontsize=32)
    sub.legend(loc='lower right', handletextpad=0.5, prop={'size': 20}) 
    #sub.text(0.95, 0.9, r'model($\theta_\mathrm{median}$)', 
    #        ha='right', va='center', transform=sub.transAxes, fontsize=20)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(''.join([UT.tex_dir(), 'figs/qaplot_abc.pdf']), bbox_inches='tight', dpi=150) 
    plt.close()
    return None 


def SHMRscatter_tduty_2panel(Mhalo=12, dMhalo=0.1, Mstar=10.5, dMstar=0.5):
    ''' Figure plotting the scatter in the Stellar to Halo Mass Relation (i.e. 
    sigma_logM*(M_h = 10^12) and sigma_logMhalo(M* = 10^10.5)) as a function of duty 
    cycle timescale (t_duty) from the ABC posteriors. 
    '''
    runs = ['randomSFH0.5gyr.sfsmf.sfsbroken', 
            'randomSFH1gyr.sfsmf.sfsbroken', 
            'randomSFH2gyr.sfsmf.sfsbroken', 
            'randomSFH5gyr.sfsmf.sfsbroken', 
            'randomSFH10gyr.sfsmf.sfsbroken']
    tduties = [0.5, 1., 2., 5., 10.]  #hardcoded
    iters = [14 for i in range(len(tduties))] # iterations of ABC
    nparticles = [1000, 1000, 1000, 1000, 1000]
    # constraints from literature 
    # constraints for sigma_logM*
    lit_siglogMs = [
            'More+(2011)', #'Yang+(2009)', # group finder results can't really be included because they use abundance matching to assign halos 
            'Leauthaud+(2012)', 
            'Tinker+(2013)', #'Reddick+(2013)', 
            'Zu+(2015)'
            ]
    lit_siglogMs_median = [0.15, #0.122, 
            0.206, 0.21, #0.20, 
            0.22]
    lit_siglogMs_upper = [0.26, #0.152, 
            0.206+0.031, 0.27, #0.23, 
            0.24]
    lit_siglogMs_lower = [0.07, #0.092, 
            0.206-0.031, 0.15, #0.17, 
            0.20] 

    # Tinker et al. (2013) for star-forming galaxies 0.2 < z < 0.48 Table 2 (COSMOS)
    # Yang et al. (2009) for blue galaxies Table 4 (SDSS DR4) 
    # More et al. (2011) SMHMR of starforming centrals (SDSS)
    # Leauthaud et al. (2012) all galaxies 0.2 < z < 0.48 (COSMOS)
    # Reddick et al. (2013) Figure 7. (constraints from conditional SMF)
    # Zu & Mandelbaum (2015) SDSS constraints on iHOD parameters (abstract & Section 7.2) 
    # Meng Gu et al. (2016) 
    
    # calculate the scatters from the ABC posteriors 
    smhmr = Obvs.Smhmr()
    sigMs = np.zeros((5, len(tduties)))
    sigMh = np.zeros((5, len(tduties)))
    for i_t, tduty in enumerate(tduties): 
        abc_dir = ''.join([UT.dat_dir(), 'abc/', runs[i_t], '/model/']) # ABC directory 
        sig_Mss, sig_Mhs = [], [] 
        for i in range(10): 
            f = pickle.load(open(''.join([abc_dir, 'model.theta', str(i), '.t', str(iters[i_t]), '.p']), 'rb'))
            subcat_sim_i = {} 
            for key in f.keys(): 
                subcat_sim_i[key] = f[key]
            
            isSF = np.where(subcat_sim_i['galtype'] == 'sf') # only SF galaxies 

            sig_ms_i = smhmr.sigma_logMstar(
                    subcat_sim_i['halo.m'][isSF], subcat_sim_i['m.star'][isSF], 
                    weights=subcat_sim_i['weights'][isSF], Mhalo=Mhalo, dmhalo=dMhalo)
            sig_mh_i = smhmr.sigma_logMhalo(
                    subcat_sim_i['halo.m'][isSF], subcat_sim_i['m.star'][isSF], 
                    weights=subcat_sim_i['weights'][isSF], Mstar=Mstar, dmstar=dMstar)
            sig_Mss.append(sig_ms_i)  
            sig_Mhs.append(sig_mh_i) 

        sig_ms_low, sig_ms_med, sig_ms_high, sig_ms_lowlow, sig_ms_hihi = np.percentile(np.array(sig_Mss), [16, 50, 84, 2.5, 97.5])
        sig_mh_low, sig_mh_med, sig_mh_high, sig_mh_lowlow, sig_mh_hihi = np.percentile(np.array(sig_Mhs), [16, 50, 84, 2.5, 97.5]) 
        sigMs[0, i_t] = sig_ms_med
        sigMs[1, i_t] = sig_ms_low
        sigMs[2, i_t] = sig_ms_high
        sigMs[3, i_t] = sig_ms_lowlow
        sigMs[4, i_t] = sig_ms_hihi
        
        sigMh[0, i_t] = sig_mh_med
        sigMh[1, i_t] = sig_mh_low
        sigMh[2, i_t] = sig_mh_high
        sigMh[3, i_t] = sig_mh_lowlow
        sigMh[4, i_t] = sig_mh_hihi
        
    ##############################
    # -- sigma_logM* -- 
    ##############################
    fig = plt.figure(figsize=(10,5)) # make figure 
    bkgd = fig.add_subplot(111, frameon=False)
    sub = fig.add_subplot(121) 
    # ABC posteriors 
    #abc_post = sub.errorbar(tduties, sigMs[0,:], yerr=[sigMs[0,:]-sigMs[1,:], sigMs[2,:]-sigMs[0,:]], fmt='.C0') 
    print tduties, sigMs[1,:]
    print tduties, sigMs[0,:]
    print tduties, sigMs[2,:]
    print tduties, sigMh[1,:]
    print tduties, sigMh[0,:]
    print tduties, sigMh[2,:]
    sub.fill_between(tduties, sigMs[3,:], sigMs[4,:], color='C0', alpha=0.1, linewidth=0) 
    sub.fill_between(tduties, sigMs[1,:], sigMs[2,:], color='C0', alpha=0.5, linewidth=0) 
    # literature 
    subplts = [] 
    marks = ['^', 's', 'o', 'v', 'x', 'D']
    #marks = ['^', 's', 'o', '8', 'x', 'D']
    for ii, tt, sig, siglow, sigup in zip(range(len(lit_siglogMs)), np.logspace(np.log10(0.7), np.log10(7), len(lit_siglogMs)), 
            lit_siglogMs_median, lit_siglogMs_lower, lit_siglogMs_upper):
        subplt = sub.errorbar([tt], [sig], yerr=[[sig-siglow], [sigup-sig]], fmt='.k', marker=marks[ii], markersize=5)
        #fmt='.C'+str(ii), markersize=10)
        subplts.append(subplt) 
    legend1 = sub.legend(subplts[:4], lit_siglogMs[:4], handletextpad=-0.5, loc=(-0.02, 0.65), prop={'size': 15})
    sub.legend(subplts[4:], lit_siglogMs[4:], loc=(0.575, 0.), handletextpad=-0.5, prop={'size': 15})
    plt.gca().add_artist(legend1)
    sub.set_xlim([0.5, 10.]) # x-axis
    sub.set_xscale('log') 
    sub.set_ylabel(r'$\sigma_{\log\,M_*} \Big(M_h = 10^{'+str(Mhalo)+r'} M_\odot \Big)$', fontsize=25) # y-axis
    sub.set_ylim([0.1, 0.4]) 
    sub.set_yticks([0.1, 0.2, 0.3, 0.4])#, 0.6]) 
    ##############################
    # -- sigma_logMh -- 
    ##############################
    # constraints for sigma_logMh
    # Mandelbaum+(2006) @ 1.5e10 Table 3 late central galaxies 
    # Conroy+(2007) table 4 all central galaxies
    # More+ (2011) Figure 9 blue central galaxies
    # van Uitert+(2011) Table 3 late central galaxies 
    # Velander+(2013) Table 4 blue central galaxies (scatter corrected so sort of like a lower-bound?)
    # Han+(2015) from Figure 9 all central galaxies. 
    lit_siglogMh = [
            'More+(2011)',
            'Mandelbaum+(2006)', #'van Uitert+(2011)', 
            'Conroy+(2007)', 
            'Velander+(2014)', 
            'Han+(2015)' 
            ]
    lit_siglogMh_median = [0.125, 0.25, 0.15, 0.122, 0.58]
    lit_siglogMh_upper = [0.055, None, None, None, None]
    lit_siglogMh_lower = [0.195, None, None, None, None]
    
    sub = fig.add_subplot(122) 
    sub.fill_between(tduties, sigMh[3,:], sigMh[4,:], color='C0', alpha=0.1, linewidth=0) 
    sub.fill_between(tduties, sigMh[1,:], sigMh[2,:], color='C0', alpha=0.5, linewidth=0, label='Hahn+(2018)') 
    subplts = [] 
    marks = ['^', '+', '*', 'v', 'd']
    for ii, tt, sig, siglow, sigup in zip(range(len(lit_siglogMh)), np.logspace(np.log10(0.7), np.log10(7), len(lit_siglogMh)), 
            lit_siglogMh_median, lit_siglogMh_lower, lit_siglogMh_upper):
        if siglow is None: 
            subplt = sub.scatter([tt], [sig], c='k', marker=marks[ii], s=60)
            subplts.append(subplt) 
        else: 
            sub.errorbar([tt], [sig], yerr=[[sig-siglow], [sigup-sig]], fmt='.k', marker=marks[ii], markersize=5)
    leg1 = sub.legend(subplts, lit_siglogMh[1:], handletextpad=-0.5, loc=(-0.02, 0.65), prop={'size': 15})
    sub.legend(loc=(0.3, 0.0), handletextpad=0.25, prop={'size': 20}) 
    plt.gca().add_artist(leg1)
    sub.set_xlim([0.5, 10.]) # x-axis
    sub.set_xscale('log') 
    sub.set_ylabel(r'$\sigma_{\log\,M_h} \Big(M_* = 10^{'+str(int(Mstar))+r'} M_\odot \Big)$', fontsize=25) # y-axis
    sub.set_ylim([0.0, 0.6]) 
    #sub.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5]) 
    sub.set_yticks([0.0, 0.2, 0.4, 0.6]) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('$t_\mathrm{duty}$ [Gyr]', labelpad=10, fontsize=22) 
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(''.join([UT.tex_dir(), 'figs/SHMRscatter_tduty.pdf']), bbox_inches='tight', dpi=150) 
    plt.close()
    return None 


def SHMRscatter_tduty(Mhalo=12, dMhalo=0.1, Mstar=10.5, dMstar=0.5):
    ''' Figure plotting the scatter in the Stellar to Halo Mass Relation (i.e. 
    sigma_logM*(M_h = 10^12) and sigma_logMhalo(M* = 10^10.5)) as a function of duty 
    cycle timescale (t_duty) from the ABC posteriors. 
    '''
    runs = ['randomSFH0.5gyr.sfsmf.sfsbroken', 
            'randomSFH1gyr.sfsmf.sfsbroken', 
            'randomSFH2gyr.sfsmf.sfsbroken', 
            'randomSFH5gyr.sfsmf.sfsbroken', 
            'randomSFH10gyr.sfsmf.sfsbroken']
    tduties = [0.5, 1., 2., 5., 10.]  #hardcoded
    iters = [14 for i in range(len(tduties))] # iterations of ABC
    nparticles = [1000, 1000, 1000, 1000, 1000]
    # constraints from literature 
    # constraints for sigma_logM*
    lit_siglogMs = [
            'More+(2011)', #'Yang+(2009)', # group finder results can't really be included because they use abundance matching to assign halos 
            'Leauthaud+(2012)', 
            'Tinker+(2013)', #'Reddick+(2013)', 
            'Zu+(2015)', 
            'Lange+(2018)*', 
            'Cao+(2019)'][::-1]
    lit_siglogMs_median = [0.15, #0.122, 
            0.206, 0.21, #0.20, 
            0.22, 
            0.2275, 
            0.33][::-1]
    lit_siglogMs_upper = [0.26, #0.152, 
            0.206+0.031, 0.27, #0.23, 
            0.24, 0.21, 0.30][::-1]
    lit_siglogMs_lower = [0.07, #0.092, 
            0.206-0.031, 0.15, #0.17, 
            0.20, 0.245, 0.36][::-1]

    # Tinker et al. (2013) for star-forming galaxies 0.2 < z < 0.48 Table 2 (COSMOS)
    # Yang et al. (2009) for blue galaxies Table 4 (SDSS DR4) 
    # More et al. (2011) SMHMR of starforming centrals (SDSS)
    # Leauthaud et al. (2012) all galaxies 0.2 < z < 0.48 (COSMOS)
    # Reddick et al. (2013) Figure 7. (constraints from conditional SMF)
    # Zu & Mandelbaum (2015) SDSS constraints on iHOD parameters (abstract & Section 7.2) 
    # Meng Gu et al. (2016) 
    
    # calculate the scatters from the ABC posteriors 
    smhmr = Obvs.Smhmr()
    sigMs = np.zeros((5, len(tduties)))
    for i_t, tduty in enumerate(tduties): 
        abc_dir = ''.join([UT.dat_dir(), 'abc/', runs[i_t], '/model/']) # ABC directory 
        sig_Mss, sig_Mhs = [], [] 
        for i in range(10): 
            f = pickle.load(open(''.join([abc_dir, 'model.theta', str(i), '.t', str(iters[i_t]), '.p']), 'rb'))
            subcat_sim_i = {} 
            for key in f.keys(): 
                subcat_sim_i[key] = f[key]
            
            isSF = np.where(subcat_sim_i['galtype'] == 'sf') # only SF galaxies 

            sig_ms_i = smhmr.sigma_logMstar(
                    subcat_sim_i['halo.m'][isSF], subcat_sim_i['m.star'][isSF], 
                    weights=subcat_sim_i['weights'][isSF], Mhalo=Mhalo, dmhalo=dMhalo)
            sig_Mss.append(sig_ms_i)  

        sig_ms_low, sig_ms_med, sig_ms_high, sig_ms_lowlow, sig_ms_hihi = np.percentile(np.array(sig_Mss), [16, 50, 84, 2.5, 97.5])
        sigMs[0, i_t] = sig_ms_med
        sigMs[1, i_t] = sig_ms_low
        sigMs[2, i_t] = sig_ms_high
        sigMs[3, i_t] = sig_ms_lowlow
        sigMs[4, i_t] = sig_ms_hihi
        
    ##############################
    # -- sigma_logM* -- 
    ##############################
    fig = plt.figure(figsize=(5,5)) # make figure 
    sub = fig.add_subplot(111) 
    # ABC posteriors 
    sub.fill_between(tduties, sigMs[3,:], sigMs[4,:], color='C0', alpha=0.1, linewidth=0) 
    sub.fill_between(tduties, sigMs[1,:], sigMs[2,:], color='C0', alpha=0.5, linewidth=0) 
    # literature 
    subplts = [] 
    marks = ['^', 's', 'o', 'v', 'x', 'D']
    #marks = ['^', 's', 'o', '8', 'x', 'D']
    for ii, tt, sig, siglow, sigup in zip(range(len(lit_siglogMs)), np.logspace(np.log10(0.7), np.log10(7), len(lit_siglogMs)), 
            lit_siglogMs_median, lit_siglogMs_lower, lit_siglogMs_upper):
        subplt = sub.errorbar([tt], [sig], yerr=[[sig-siglow], [sigup-sig]], fmt='.k', marker=marks[ii], markersize=8)
        #fmt='.C'+str(ii), markersize=10)
        subplts.append(subplt) 
    #legend1 = sub.legend(subplts[:2], lit_siglogMs[:2], handletextpad=-0.5, loc=(-0.02, 0.7), prop={'size': 15})
    #sub.legend(subplts[2:], lit_siglogMs[2:], loc=(0.575, 0.), handletextpad=-0.5, prop={'size': 15})
    legend1 = sub.legend(subplts[:3], lit_siglogMs[:3], handletextpad=-0.5, loc=(-0.02, 0.05), prop={'size': 15})
    sub.legend(subplts[3:], lit_siglogMs[3:], loc=(0.45, 0.72), handletextpad=-0.5, prop={'size': 15})
    plt.gca().add_artist(legend1)
    sub.set_xlim([0.5, 10.]) # x-axis
    sub.set_xscale('log') 
    sub.set_ylabel(r'$\sigma_{\log\,M_*} \Big(M_h = 10^{'+str(Mhalo)+r'} M_\odot \Big)$', fontsize=25) # y-axis
    sub.set_ylim([0.1, 0.4]) 
    sub.set_yticks([0.1, 0.2, 0.3, 0.4]) 
    sub.set_xlabel('$t_\mathrm{duty}$ [Gyr]', labelpad=5, fontsize=22) 
    fig.savefig(''.join([UT.tex_dir(), 'figs/SHMRscatter_tduty.pdf']), bbox_inches='tight', dpi=150) 
    plt.close()
    return None 


def SHMRscatter_tduty_v2(Mhalo=12, dMhalo=0.1):
    ''' Figure comparing the scatter in the Stellar to Halo Mass Relation (i.e. 
    sigma_logM*(M_h = 10^12) as a function of duty cycle timescale (t_duty) from 
    the ABC posteriors to observations and different simulations. 
    '''
    # calculate the scatters from the ABC posteriors 
    runs = ['randomSFH0.1gyr.sfsmf.sfsbroken', 
            'randomSFH0.5gyr.sfsmf.sfsbroken', 
            'randomSFH1gyr.sfsmf.sfsbroken', 
            'randomSFH2gyr.sfsmf.sfsbroken', 
            'randomSFH5gyr.sfsmf.sfsbroken', 
            'randomSFH10gyr.sfsmf.sfsbroken']
    tduties = [0.1, 0.5, 1., 2., 5., 10.]  #hardcoded
    iters = [14 for i in range(len(tduties))] # iterations of ABC
    nparticles = [1000, 1000, 1000, 1000, 1000, 1000]

    smhmr = Obvs.Smhmr()
    sigMs = np.zeros((5, len(tduties)))
    for i_t, tduty in enumerate(tduties): 
        abc_dir = ''.join([UT.dat_dir(), 'abc/', runs[i_t], '/model/']) # ABC directory 
        sig_Mss, sig_Mhs = [], [] 
        for i in range(100): 
            f = pickle.load(open(''.join([abc_dir, 'model.theta', str(i), '.t', str(iters[i_t]), '.p']), 'rb'))
            subcat_sim_i = {} 
            for key in f.keys(): 
                subcat_sim_i[key] = f[key]
            
            isSF = np.where(subcat_sim_i['galtype'] == 'sf') # only SF galaxies 

            sig_ms_i = smhmr.sigma_logMstar(
                    subcat_sim_i['halo.m'][isSF], subcat_sim_i['m.star'][isSF], 
                    weights=subcat_sim_i['weights'][isSF], Mhalo=Mhalo, dmhalo=dMhalo)
            sig_Mss.append(sig_ms_i)  

        sig_ms_low, sig_ms_med, sig_ms_high, sig_ms_lowlow, sig_ms_hihi = np.percentile(np.array(sig_Mss), [16, 50, 84, 2.5, 97.5])
        sigMs[0, i_t] = sig_ms_med
        sigMs[1, i_t] = sig_ms_low
        sigMs[2, i_t] = sig_ms_high
        sigMs[3, i_t] = sig_ms_lowlow
        sigMs[4, i_t] = sig_ms_hihi
        print('t_duty = %.1f --- sigma_{M*|Mh=10^%.1f Msun} = %f (+/- %f, %f)' % 
                (tduty, Mhalo, sig_ms_med, sig_ms_high-sig_ms_med, sig_ms_med - sig_ms_low))
        
    ##############################
    # sigma_logM* vs observaitons 
    ##############################
    fig = plt.figure(figsize=(10,5)) # make figure 
    sub = fig.add_subplot(121) 
    # abc posteriors 
    sub.fill_between(tduties, sigMs[3,:], sigMs[4,:], color='C0', alpha=0.1, linewidth=0) 
    sub.fill_between(tduties, sigMs[1,:], sigMs[2,:], color='C0', alpha=0.5, linewidth=0) 

    # sigma_logM* constraints from literature 
    # Tinker et al. (2013) for star-forming galaxies 0.2 < z < 0.48 (COSMOS)
    # More et al. (2011) SMHMR of starforming centrals (SDSS)
    # Leauthaud et al. (2012) all galaxies 0.2 < z < 0.48 (COSMOS)
    # Reddick et al. (2013) Figure 7. (constraints from conditional SMF)
    # Zu & Mandelbaum (2015) SDSS constraints on iHOD parameters
    slm_cao2019 = [0.30, 0.33, 0.36] 
    plt_cao = sub.fill_between(np.linspace(0., 10., 100), np.repeat(slm_cao2019[0], 100), np.repeat(slm_cao2019[2], 100), 
            facecolor='none', hatch='///', edgecolor='k', linewidth=0., zorder=1)
    sub.text(1., slm_cao2019[1], 'Cao et al.(2019)', ha='center', va='center', fontsize=15)
    
    lit_names       = ['Leauthaud+(2012)', 'Reddick+(2013)', 'Tinker+(2017)', 'Zu+(2015)', 'Lange+(2018)*']
    slm_lit         = np.zeros((len(lit_names), 3)) 
    slm_lit[:,0]    = np.array([0.206-0.031, 0.185, 0.16, 0.20, 0.245]) 
    slm_lit[:,1]    = np.array([0.206, 0.21, 0.18, 0.22, 0.2275]) 
    slm_lit[:,2]    = np.array([0.206+0.031, 0.235, 0.19, 0.24, 0.21]) 
    slm_lit_tot     = np.array([slm_lit[:,0].min(), 0, slm_lit[:,2].max()]) 
    plt_lit = sub.fill_between(np.linspace(0., 10., 100), 
            np.repeat(slm_lit_tot[0], 100), np.repeat(slm_lit_tot[2], 100), 
            facecolor='k', alpha=0.25, linewidth=0., zorder=1)
    sub.text(1., 0.5*(slm_lit_tot[0] + slm_lit_tot[2]), 'literature (see Table 1)', ha='center', va='center', fontsize=15) 
    sub.set_xlim([0.1, 10.]) # x-axis
    sub.set_xscale('log') 
    sub.set_ylabel(r'$\sigma_{M_*|M_h}$ at $M_h = 10^{'+str(Mhalo)+r'} M_\odot$', fontsize=25) # y-axis
    sub.set_ylim([0.1, 0.4]) 
    sub.set_yticks([0.1, 0.2, 0.3, 0.4]) 
    #sub.set_xlabel('$t_\mathrm{duty}$ [Gyr]', labelpad=5, fontsize=22) 

    ################################################
    # sigma_logM* vs siulations 
    ################################################
    sub2 = fig.add_subplot(122) 
    # abc posteriors 
    abs_post = sub2.fill_between(tduties, sigMs[3,:], sigMs[4,:], color='C0', alpha=0.1, linewidth=0)
    sub2.fill_between(tduties, sigMs[1,:], sigMs[2,:], color='C0', alpha=0.5, linewidth=0,
            label='Hahn+(2019) posteriors') 
    sim_plts = [] 
    # hydro sims
    sim_plt = sub2.fill_between(np.linspace(0., 10., 100), np.repeat(0.16, 100), np.repeat(0.22, 100), 
            facecolor='none', hatch='...', edgecolor='k', linewidth=0., zorder=1)
    sim_plts.append(sim_plt) 
    # SAMs
    sim_plt = sub2.fill_between(np.linspace(0., 10., 100), np.repeat(0.30, 100), np.repeat(0.37, 100), 
            facecolor='none', hatch='X', edgecolor='k', linewidth=0.)
    sim_plts.append(sim_plt) 
    # universe machine
    sub2.plot(np.linspace(0., 10., 100), np.repeat(0.28, 100), c='k', ls='--')
    sub2.text(0.975, 0.59, 'UniverseMachine', ha='right', va='top', transform=sub2.transAxes, fontsize=12)
    # Abramson+(2016)
    sim_plt = sub2.plot(np.linspace(0., 10., 100), np.repeat(0.33, 100), c='k', ls=':')
    #sim_plts.append(sim_plt) 
    r = Rectangle((0.11, 0.315), 0.9, 0.0125, linewidth=0., fc='white') 
    plt.gca().add_patch(r)
    sub2.text(0.025, 0.76, 'Abramson+(2016)w/AM', ha='left', va='top', transform=sub2.transAxes, fontsize=12)

    sub2.text(0.5, 0.3, 'hydro. sims', ha='center', va='center', transform=sub2.transAxes, fontsize=15)
    r = Rectangle((0.5, 0.183), 1.5, 0.017, linewidth=0., fc='white') 
    plt.gca().add_patch(r)
    sub2.text(0.5, 0.825, 'semi-analytic models', ha='center', va='center', transform=sub2.transAxes, fontsize=15)
    r = Rectangle((0.29, 0.34), 3., 0.018, linewidth=0., fc='white') 
    plt.gca().add_patch(r)
    sub2.legend(loc=(0.2, 0.025), handletextpad=0.4, frameon=False, fontsize=15) 
    sub2.set_xlim([0.1, 10.]) # x-axis
    sub2.set_xscale('log') 
    sub2.set_ylim([0.1, 0.4]) 
    sub2.set_yticks([0.1, 0.2, 0.3, 0.4])#, 0.6]) 
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('$t_\mathrm{duty}$ [Gyr]', labelpad=10, fontsize=22) 
    fig.subplots_adjust(wspace=0.2)
    fig.savefig(''.join([UT.tex_dir(), 'figs/SHMRscatter_tduty_v2.pdf']), bbox_inches='tight', dpi=150) 
    plt.close()
    return None 


def Mhacc_dSFR(runs, T): 
    '''
    '''
    # calculate Mhalo history and dSFR 
    mhacc_run, dsfr_run = [], [] 
    for i_run, run in enumerate(runs): 
        abcout = ABC.readABC(run, T)
        theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) for i in range(len(abcout['theta'][0]))] # median theta 
        tt = ABC._model_theta(run, theta_med)
        dsfrs = np.array([10.]) 
        while np.abs(dsfrs).max() > 1.: 
            np.random.RandomState(0)
            # summary statistics of model 
            subcat_sim = ABC.model(run, theta_med, nsnap0=15, downsampled='20') 
            n_halo = len(subcat_sim['m.star'])

            if i_run == 0: 
                mhlim = ((subcat_sim['galtype'] == 'sf') & (subcat_sim['weights'] > 0.) & 
                        (subcat_sim['halo.m'] > 12.) & (subcat_sim['halo.m'] < 12.1)) 
                i_halo = np.random.choice(np.arange(n_halo)[mhlim], 2, replace=False) 

            # get Mh accretion history 
            mhacc = np.zeros((len(i_halo), 20))
            mhacc[:,0] = subcat_sim['halo.m'][i_halo]
            for i in range(2, 21): 
                mhacc[:,i-1] = subcat_sim['halo.m.snap'+str(i)][i_halo]

            msacc = np.zeros((len(i_halo), 15))
            msacc[:,0] = subcat_sim['m.star'][i_halo]
            dsfrs = np.zeros((len(i_halo), 15))
            dsfrs[:,0] = subcat_sim['sfr'][i_halo] - SFH.SFR_sfms(subcat_sim['m.star'][i_halo], UT.z_nsnap(1), tt['sfms'])
            for i in range(2, 16): 
                if i != 15: 
                    msacc[:,i-1] = subcat_sim['m.star.snap'+str(i)][i_halo]
                    dsfrs[:,i-1] = subcat_sim['sfr.snap'+str(i)][i_halo] - \
                            SFH.SFR_sfms(subcat_sim['m.star.snap'+str(i)][i_halo], UT.z_nsnap(i), tt['sfms'])
                else: 
                    msacc[:,i-1] = subcat_sim['m.star0'][i_halo]
                    dsfrs[:,i-1] = subcat_sim['sfr0'][i_halo] - \
                            SFH.SFR_sfms(subcat_sim['m.star0'][i_halo], UT.z_nsnap(i), tt['sfms'])
        mhacc_run.append(mhacc) 
        dsfr_run.append(dsfrs)

    fig = plt.figure(figsize=(6,8))
    sub = fig.add_subplot(311)
    for ii in range(mhacc.shape[0]): 
        sub.plot(UT.t_nsnap(np.arange(1,21)), # - UT.tdyn_t(UT.t_nsnap(np.arange(1,21))), 
                10**(mhacc[ii,:]-mhacc[ii,0]), c='C'+str(ii))
    #for ii in range(msacc.shape[0]): 
    #    sub.plot(UT.t_nsnap(np.arange(1,16)), 10**(msacc[ii,:]-msacc[ii,0])-0.5, c='C'+str(ii), ls=":")

    sub.vlines(UT.t_nsnap(5), -1., 1., color='k', linestyle=':', linewidth=2) 
    sub.vlines(UT.t_nsnap(5) - UT.tdyn_t(UT.t_nsnap(5)), -1., 1., color='k', linestyle=':', linewidth=1.25) 
    #sub.fill_between([UT.t_nsnap(5), UT.t_nsnap(5) - UT.tdyn_t(UT.t_nsnap(5))], [0.,0.], [1.,1.], 
    #        linewidth=0., color='k', alpha=0.15)  
    sub.text(0.58, 0.1, r"$t_\mathrm{dyn}$", ha='left', va='bottom', transform=sub.transAxes, fontsize=15)
    sub.annotate(s='', 
            xy=(UT.t_nsnap(5), 0.1), 
            xytext=(UT.t_nsnap(5) - UT.tdyn_t(UT.t_nsnap(5)), 0.1), 
            arrowprops=dict(arrowstyle='<->'))

    sub.text(0.025, 0.925, r"$M_h(z{=}0.05) \sim 10^{12} M_\odot$", ha='left', va='top', 
            transform=sub.transAxes, fontsize=17)
    sub.set_xlim([UT.t_nsnap(20), UT.t_nsnap(1)]) 
    sub.set_ylabel('$M_h / M_h(z{=}0.05)$', fontsize=20)
    sub.set_ylim([0., 1.]) 

    for i_run, run in enumerate(runs): 
        mhacc = mhacc_run[i_run] 
        dsfrs = dsfr_run[i_run]
        sub = fig.add_subplot(3,1,i_run+2)
        sub.fill_between([0., 20.], [-0.3, -0.3], [0.3, 0.3], linewidth=0., color='k', alpha=0.15)  
        for ii in range(dsfrs.shape[0]): 
            sub.plot(UT.t_nsnap(np.arange(1,16)), dsfrs[ii,:], c='C'+str(ii))
        sub.plot([0., 15.], [0., 0.,], c='k', lw=2, ls='--') 
        sub.vlines(UT.t_nsnap(5), -1., 1., color='k', linestyle=':', linewidth=2) 
        # parse run name 
        str_r = run.split('_')[1].split('abias')[-1]
        sub.text(0.025, 0.925, r"$r="+str_r+"$", ha='left', va='top', transform=sub.transAxes, fontsize=20)

        if i_run != 0: sub.set_xlabel('$t_\mathrm{cosmic}$ [Gyr]', fontsize=25) 
        sub.set_xlim([UT.t_nsnap(20), UT.t_nsnap(1)]) 
        sub.set_ylabel('$\Delta \log\,\mathrm{SFR}$', fontsize=20)
        sub.set_ylim([-1., 1.]) 
        sub.set_yticks([-1., -0.5, 0., 0.5, 1.]) 
        #if i_run == 0: sub.set_title("Assembly Bias $r="+str_r+"$", fontsize=20) 
        #else: sub.set_title("$r="+str_r+"$", fontsize=20) 

        fig.subplots_adjust(wspace=0.2, hspace=0.225)
        fig.savefig(''.join([UT.tex_dir(), 'figs/Mhacc_dSFR.pdf']), 
                bbox_inches='tight', dpi=150) 
        plt.close()
    return None 


def SHMRscatter_tduty_abias(Mhalo=12, dMhalo=0.5, Mstar=10.5, dMstar=0.5):
    ''' Figure plotting the scatter in the Stellar to Halo Mass Relation (i.e. 
    sigma_logM*(M_h = 10^12) and sigma_logMhalo(M* = 10^10.5)) as a function of duty 
    cycle timescale (t_duty) from the ABC posteriors. 
    '''
    # constraints from literature 
    # constraints for sigma_logM*
    lit_siglogMs = [
            'More+(2011)', 
            'Yang+(2009)', 
            'Leauthaud+(2012)', 
            'Tinker+(2013)', 
            'Reddick+(2013)', 
            'Zu+(2015)'
            ]
    lit_siglogMs_median = [0.15, 0.122, 0.206, 0.21, 0.20, 0.22]
    lit_siglogMs_upper = [0.26, 0.152, 0.206+0.031, 0.27, 0.23, 0.24]
    lit_siglogMs_lower = [0.07, 0.092, 0.206-0.031, 0.15, 0.17, 0.20] 
    # Tinker et al. (2013) for star-forming galaxies 0.2 < z < 0.48 (COSMOS)
    # More et al. (2011) SMHMR of starforming centrals (SDSS)
    # Leauthaud et al. (2012) all galaxies 0.2 < z < 0.48 (COSMOS)
    # Reddick et al. (2013) Figure 7. (constraints from conditional SMF)
    # Zu & Mandelbaum (2015) SDSS constraints on iHOD parameters
    # Meng Gu et al. (2016) 
    
    # constraints for sigma_logMh
    lit_siglogMh = [
            'Mandelbaum+(2006)', #2.75673e+10, 1.18497e+1 2.75971e+10, 1.21469e+1 2.76147e+10, 1.23217e+1
            'Velander+(2013)', # 2.10073e+10, 1.21993e+1 2.12155e+10, 1.24091e+1 2.10347e+10, 1.25577e+1
            'Han+(2015)' # 2.53944e+10, 1.17972e+1 2.54275e+10, 1.21556e+1 2.54615e+10, 1.25227e+1
            ]
    lit_siglogMh_median = [0.47, 0.36, 0.72]
    
    # make figure 
    fig = plt.figure(figsize=(10,5)) 
    bkgd = fig.add_subplot(111, frameon=False)
    sub1 = fig.add_subplot(121)
    sub2 = fig.add_subplot(122) 

    # calculate the scatters from the ABC posteriors 
    smhmr = Obvs.Smhmr()
    tscales = ['0.5', '1', '2', '5', '10']
    for i_a, abias in enumerate([0., 0.5, 0.99]): 
        if abias > 0.:
            #runs = ['rSFH_r'+str(abias)+'_tdyn_'+str(tt)+'gyr' for tt in [0.5, 1, 2, 5]]
            runs = ['rSFH_abias'+str(abias)+'_'+tt+'gyr.sfsmf.sfsbroken' for tt in tscales]
            if abias == 0.99: mark='^'
            else: mark='s'
            ms=4
        else: 
            runs = ['randomSFH'+tt+'gyr.sfsmf.sfsbroken' for tt in tscales]
            mark=None
            ms=None
        tduties = [0.5, 1., 2., 5., 9.75]  #hardcoded
        iters = [14, 14, 14, 14, 14] # iterations of ABC
        nparticles = [1000, 1000, 1000, 1000, 1000]

        sigMs = np.zeros((5, len(tduties)))
        sigMh = np.zeros((5, len(tduties)))
        for i_t, tduty in enumerate(tduties): 
            abc_dir = UT.dat_dir()+'abc/'+runs[i_t]+'/model/' # ABC directory 
            sig_Mss, sig_Mhs = [], [] 
            for i in range(10): 
                f = pickle.load(open(''.join([abc_dir, 'model.theta', str(i), '.t', str(iters[i_t]), '.p']), 'rb'))
                subcat_sim_i = {} 
                for key in f.keys(): 
                    subcat_sim_i[key] = f[key]
                
                isSF = np.where(subcat_sim_i['galtype'] == 'sf') # only SF galaxies 

                sig_ms_i = smhmr.sigma_logMstar(
                        subcat_sim_i['halo.m'][isSF], subcat_sim_i['m.star'][isSF], 
                        weights=subcat_sim_i['weights'][isSF], Mhalo=Mhalo, dmhalo=dMhalo)
                sig_mh_i = smhmr.sigma_logMhalo(
                        subcat_sim_i['halo.m'][isSF], subcat_sim_i['m.star'][isSF], 
                        weights=subcat_sim_i['weights'][isSF], Mstar=Mstar, dmstar=dMstar)
                sig_Mss.append(sig_ms_i)  
                sig_Mhs.append(sig_mh_i) 
            sig_ms_low, sig_ms_med, sig_ms_high, sig_ms_lowlow, sig_ms_hihi = np.percentile(np.array(sig_Mss), [16, 50, 84, 2.5, 97.5])
            sig_mh_low, sig_mh_med, sig_mh_high, sig_mh_lowlow, sig_mh_hihi = np.percentile(np.array(sig_Mhs), [16, 50, 84, 2.5, 97.5]) 
            sigMs[0, i_t] = sig_ms_med
            sigMs[1, i_t] = sig_ms_low
            sigMs[2, i_t] = sig_ms_high
            sigMs[3, i_t] = sig_ms_lowlow
            sigMs[4, i_t] = sig_ms_hihi
            
            sigMh[0, i_t] = sig_mh_med
            sigMh[1, i_t] = sig_mh_low
            sigMh[2, i_t] = sig_mh_high
            sigMh[3, i_t] = sig_mh_lowlow
            sigMh[4, i_t] = sig_mh_hihi
        # ABC posteriors 
        #abc_post1 = sub1.errorbar(10**(np.log10(tduties)+0.01*i_a), sigMs[0,:], 
        #        yerr=[sigMs[0,:]-sigMs[1,:], sigMs[2,:]-sigMs[0,:]], 
        #        fmt='.k', marker=mark, markersize=ms) 
        #sub1.fill_between(tduties, sigMs[3,:], sigMs[4,:], color='C'+str(i_a), alpha=0.1, linewidth=0) 
        sub1.fill_between(tduties, sigMs[1,:], sigMs[2,:], color='C'+str(i_a), alpha=0.5, linewidth=0) 
        #abc_post2 = sub2.errorbar(10**(np.log10(tduties)+0.01*i_a), sigMh[0,:], 
        #        yerr=[sigMh[0,:]-sigMh[1,:], sigMh[2,:]-sigMh[0,:]], 
        #        fmt='.k', marker=mark, markersize=ms) 
        #sub2.fill_between(tduties, sigMh[3,:], sigMh[4,:], color='C'+str(i_a), alpha=0.1, linewidth=0) 
        sub2.fill_between(tduties, sigMh[1,:], sigMh[2,:], color='C'+str(i_a), alpha=0.5, linewidth=0) 

    # literature 
    subplts = [] 
    marks = ['^', 's', 'o', 'v', '8', 'x', 'D']
    for ii, tt, sig, siglow, sigup in zip(range(len(lit_siglogMs)), np.logspace(np.log10(0.7), np.log10(7), len(lit_siglogMs)), 
            lit_siglogMs_median, lit_siglogMs_lower, lit_siglogMs_upper):
        subplt = sub1.errorbar([tt], [sig], yerr=[[sig-siglow], [sigup-sig]], fmt='.k', marker=marks[ii], markersize=5) #fmt='.C'+str(ii), markersize=10)
        subplts.append(subplt) 
    legend1 = sub1.legend(subplts[:4], lit_siglogMs[:4], handletextpad=-0.5, loc=(-0.02, 0.65), prop={'size': 15})
    sub1.legend(subplts[4:], lit_siglogMs[4:], loc=(0.425, 0.), handletextpad=-0.5, prop={'size': 15})
    sub1.add_artist(legend1)
    sub1.set_xlim([0.5, 10.]) # x-axis
    sub1.set_xscale('log') 
    sub1.set_ylabel(r'$\sigma_{M_*} \Big(M_\mathrm{halo} = 10^{'+str(Mhalo)+r'} M_\odot \Big)$', fontsize=20) # y-axis
    sub1.set_ylim([0.1, 0.4]) 
    sub1.set_yticks([0.1, 0.2, 0.3, 0.4])#, 0.6]) 
    # ABC posteriors 
    #subplts = [] 
    #for ii, tt, sig in zip(range(len(lit_siglogMh)), np.logspace(np.log10(0.7), np.log10(7), len(lit_siglogMh)), lit_siglogMh_median):
    #    subplt = sub2.plot([tt/1.03, tt*1.03], [sig, sig], color='C'+str(ii))#, yerr=0.02, uplims=True)
    #    subplt = sub2.errorbar([tt], [sig], yerr=0.02, uplims=True, color='C'+str(ii))
    #    subplts.append(subplt) 
    #sub2.legend(subplts, lit_siglogMh, loc='lower right', markerscale=4, prop={'size': 15})
    #abc_post1 = sub2.errorbar([0], [0], yerr=[0.1], fmt='.k', marker=None, markersize=None) 
    #abc_post2 = sub2.errorbar([0], [0], yerr=[0.1], fmt='.k', marker='s', markersize=3) 
    #abc_post3 = sub2.errorbar([0], [0], yerr=[0.1], fmt='.k', marker='^', markersize=3) 
    abc_post1 = sub2.fill_between([0], [0], [0.1], color='C0', label='no assembly bias') 
    abc_post2 = sub2.fill_between([0], [0], [0.1], color='C1', label='$r=0.5$') 
    abc_post3 = sub2.fill_between([0], [0], [0.1], color='C2', label='$r=0.99$') 
    sub2.text(0.05, 0.95, 'Hahn+(2018) posteriors', ha='left', va='top', transform=sub2.transAxes, fontsize=20)
    sub2.legend(loc=(0.05, 0.62), handletextpad=0.4, fontsize=15) 
    sub2.set_xlim([0.5, 10.]) # x-axis
    sub2.set_xscale('log') 
    sub2.set_ylabel(r'$\sigma_{M_\mathrm{halo}} \Big(M_* = 10^{'+str(Mstar)+r'} M_\odot \Big)$', fontsize=20) # y-axis
    sub2.set_ylim([0.1, 0.5]) 
    sub2.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5]) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('$t_\mathrm{duty}$ [Gyr]', labelpad=10, fontsize=22) 
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(''.join([UT.tex_dir(), 'figs/SHMRscatter_tduty_abias.pdf']), 
            bbox_inches='tight', dpi=150) 
    plt.close()
    return None 


def SHMRscatter_tduty_abias_v2(Mhalo=12, dMhalo=0.5, Mstar=10.5, dMstar=0.5):
    ''' Figure plotting the scatter in the Stellar to Halo Mass Relation (i.e. 
    sigma_logM*(M_h = 10^12) and sigma_logMhalo(M* = 10^10.5)) as a function of duty 
    cycle timescale (t_duty) from the ABC posteriors. 
    '''
    fig = plt.figure(figsize=(10,5)) 
    bkgd = fig.add_subplot(111, frameon=False)
    sub1 = fig.add_subplot(121)
    sub2 = fig.add_subplot(122) 
    # calculate the scatters from the ABC posteriors 
    smhmr = Obvs.Smhmr()
    tscales = ['0.1', '0.5', '1', '2', '5', '10']
    for i_a, abias in enumerate([0., 0.5, 0.99]): 
        if abias > 0.:
            #runs = ['rSFH_r'+str(abias)+'_tdyn_'+str(tt)+'gyr' for tt in [0.5, 1, 2, 5]]
            runs = ['rSFH_abias'+str(abias)+'_'+tt+'gyr.sfsmf.sfsbroken' for tt in tscales]
            if abias == 0.99: mark='^'
            else: mark='s'
            ms=4
        else: 
            runs = ['randomSFH'+tt+'gyr.sfsmf.sfsbroken' for tt in tscales]
            mark=None
            ms=None
        tduties = [0.1, 0.5, 1., 2., 5., 9.75]  #hardcoded
        iters = [14, 14, 14, 14, 14, 14] # iterations of ABC
        nparticles = [1000, 1000, 1000, 1000, 1000, 1000]

        sigMs = np.zeros((5, len(tduties)))
        for i_t, tduty in enumerate(tduties): 
            abc_dir = UT.dat_dir()+'abc/'+runs[i_t]+'/model/' # ABC directory 
            sig_Mss, sig_Mhs = [], [] 
            for i in range(100): 
                f = pickle.load(open(''.join([abc_dir, 'model.theta', str(i), '.t', str(iters[i_t]), '.p']), 'rb'))
                subcat_sim_i = {} 
                for key in f.keys(): 
                    subcat_sim_i[key] = f[key]
                
                isSF = np.where(subcat_sim_i['galtype'] == 'sf') # only SF galaxies 

                sig_ms_i = smhmr.sigma_logMstar(
                        subcat_sim_i['halo.m'][isSF], subcat_sim_i['m.star'][isSF], 
                        weights=subcat_sim_i['weights'][isSF], Mhalo=Mhalo, dmhalo=dMhalo)
                sig_mh_i = smhmr.sigma_logMhalo(
                        subcat_sim_i['halo.m'][isSF], subcat_sim_i['m.star'][isSF], 
                        weights=subcat_sim_i['weights'][isSF], Mstar=Mstar, dmstar=dMstar)
                sig_Mss.append(sig_ms_i)  
                sig_Mhs.append(sig_mh_i) 
            sig_ms_low, sig_ms_med, sig_ms_high, sig_ms_lowlow, sig_ms_hihi = np.percentile(np.array(sig_Mss), [16, 50, 84, 2.5, 97.5])
            sigMs[0, i_t] = sig_ms_med
            sigMs[1, i_t] = sig_ms_low
            sigMs[2, i_t] = sig_ms_high
            sigMs[3, i_t] = sig_ms_lowlow
            sigMs[4, i_t] = sig_ms_hihi
            print('t_duty = %.1f, r = %.1f --- sigma_M*|Mh = %f (+/- %f, %f)' % 
                    (tduty, abias, sig_ms_med, sig_ms_high-sig_ms_med, sig_ms_med - sig_ms_low))
        # ABC posteriors 
        sub1.fill_between(tduties, sigMs[1,:], sigMs[2,:], color='C'+str(i_a), alpha=0.5, linewidth=0) 
        sub2.fill_between(tduties, sigMs[1,:], sigMs[2,:], color='C'+str(i_a), alpha=0.5, linewidth=0, zorder=2) 
    
    # sigma_logM* constraints from literature 
    # Tinker et al. (2013) for star-forming galaxies 0.2 < z < 0.48 (COSMOS)
    # More et al. (2011) SMHMR of starforming centrals (SDSS)
    # Leauthaud et al. (2012) all galaxies 0.2 < z < 0.48 (COSMOS)
    # Reddick et al. (2013) Figure 7. (constraints from conditional SMF)
    # Zu & Mandelbaum (2015) SDSS constraints on iHOD parameters
    slm_cao2019 = [0.30, 0.33, 0.36] 
    plt_cao = sub1.fill_between(np.linspace(0., 10., 100), np.repeat(slm_cao2019[0], 100), np.repeat(slm_cao2019[2], 100), 
            facecolor='none', hatch='///', edgecolor='k', linewidth=0., zorder=1)
    sub1.text(1., slm_cao2019[1], 'Cao et al.(2019)', ha='center', va='center', fontsize=15)
    
    lit_names       = ['Leauthaud+(2012)', 'Reddick+(2013)', 'Tinker+(2017)', 'Zu+(2015)', 'Lange+(2018)*']
    slm_lit         = np.zeros((len(lit_names), 3)) 
    slm_lit[:,0]    = np.array([0.206-0.031, 0.185, 0.16, 0.20, 0.245]) 
    slm_lit[:,1]    = np.array([0.206, 0.21, 0.18, 0.22, 0.2275]) 
    slm_lit[:,2]    = np.array([0.206+0.031, 0.235, 0.19, 0.24, 0.21]) 
    slm_lit_tot     = np.array([slm_lit[:,0].min(), 0, slm_lit[:,2].max()]) 
    plt_lit = sub1.fill_between(np.linspace(0., 10., 100), 
            np.repeat(slm_lit_tot[0], 100), np.repeat(slm_lit_tot[2], 100), 
            facecolor='k', alpha=0.25, linewidth=0., zorder=1)
    leg1 = sub1.legend([plt_lit], ['literature (see Table 1)'], loc='lower left', handletextpad=0.5, prop={'size': 15})
    sub1.add_artist(leg1)
    sub1.set_xlim([0.1, 10.]) # x-axis
    sub1.set_xscale('log') 
    sub1.set_ylabel(r'$\sigma_{M_*|M_h}$ at $M_{h} = 10^{'+str(Mhalo)+r'} M_\odot$', fontsize=25) # y-axis
    sub1.set_ylim([0.1, 0.4]) 
    sub1.set_yticks([0.1, 0.2, 0.3, 0.4])#, 0.6]) 

    ################################################
    # sigma_logM* for simulations 
    ################################################
    # constraints for sigma_logMh
    # hydro sims
    plt_hyd = sub2.fill_between(np.linspace(0., 10., 100), np.repeat(0.16, 100), np.repeat(0.22, 100), 
            facecolor='none', hatch='...', edgecolor='k', linewidth=0., zorder=1)
    # SAMs
    sub2.fill_between(np.linspace(0., 10., 100), np.repeat(0.30, 100), np.repeat(0.37, 100), 
            facecolor='none', hatch='X', edgecolor='k', linewidth=0.)
    plt_sam = sub2.fill_between(np.linspace(0., 10., 100), np.repeat(0., 100), np.repeat(0., 100), 
            facecolor='none', hatch='XX', edgecolor='k', linewidth=0.)
    # universe machine
    sub2.plot(np.linspace(0., 10., 100), np.repeat(0.28, 100), c='k', ls='--')
    sub2.text(0.975, 0.59, 'UniverseMachine', ha='right', va='top', transform=sub2.transAxes, fontsize=12)

    leg1 = sub2.legend([plt_hyd, plt_sam], ['hydro. sims', 'semi-analytic'], 
            loc='lower right', handletextpad=0.5, prop={'size': 15})

    abc_post1 = sub2.fill_between([0], [0], [0.1], color='C0', label='$r=0$')#'no assembly bias') 
    abc_post2 = sub2.fill_between([0], [0], [0.1], color='C1', label='$r=0.5$') 
    abc_post3 = sub2.fill_between([0], [0], [0.1], color='C2', label='$r=0.99$') 
    sub2.text(0.04, 0.965, 'Hahn+(2018) posteriors', ha='left', va='top', transform=sub2.transAxes, fontsize=16)
    r = Rectangle((0.2, 0.308), 0.5, 0.014, linewidth=0., fc='white') 
    plt.gca().add_patch(r)
    r = Rectangle((0.2, 0.33), 0.44, 0.014, linewidth=0., fc='white') 
    plt.gca().add_patch(r)
    #r = Rectangle((0.8, 0.35), 2.3, 0.02, linewidth=0., fc='white') 
    r = Rectangle((0.2, 0.352), 0.3, 0.014, linewidth=0., fc='white') 
    plt.gca().add_patch(r)
    leg2 = sub2.legend(loc=(0.02, 0.67), handletextpad=0.4, frameon=False, fontsize=15) 
    sub2.add_artist(leg1)
    sub2.add_artist(leg2)
    sub2.set_xlim([0.1, 10.]) # x-axis
    sub2.set_xscale('log') 
    #sub2.set_ylabel(r'$\sigma_{M_\mathrm{halo}} \Big(M_* = 10^{'+str(Mstar)+r'} M_\odot \Big)$', fontsize=20) # y-axis
    sub2.set_ylim([0.1, 0.4]) 
    sub2.set_yticks([0.1, 0.2, 0.3, 0.4])#, 0.6]) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('$t_\mathrm{duty}$ [Gyr]', labelpad=10, fontsize=22) 
    fig.subplots_adjust(wspace=0.2)
    fig.savefig(''.join([UT.tex_dir(), 'figs/SHMRscatter_tduty_abias2.pdf']), 
            bbox_inches='tight', dpi=150) 
    plt.close()
    return None 


def SHMRscatter_tduty_abias_contour(Mhalo=12, dMhalo=0.1, niter=14): 
    '''     
    '''
    tduties = [0.1, 0.5, 1, 2, 5, 10]
    r_abias = [0., 0.5, 0.99]
    
    smhmr = Obvs.Smhmr()

    # get sigma_M*(log M_h = 12.) for a grid of tduty and r
    sigma_grid = np.zeros((len(tduties), len(r_abias)))
    sigma_grid_low = np.zeros((len(tduties), len(r_abias)))
    sigma_grid_high = np.zeros((len(tduties), len(r_abias)))
    for i, tduty in enumerate(tduties): 
        for j, r in enumerate(r_abias): 
            if r > 0: 
                run = 'rSFH_abias'+str(r)+'_'+str(tduty)+'gyr.sfsmf.sfsbroken'
            else: 
                run = 'randomSFH'+str(tduty)+'gyr.sfsmf.sfsbroken'

            abc_dir = UT.dat_dir()+'abc/'+run+'/model/' # ABC directory 
            sig_Mss, sig_Mhs = [], [] 
            for ii in range(100): 
                f = pickle.load(open(''.join([abc_dir, 'model.theta', str(ii), '.t', str(niter), '.p']), 'rb'))
                subcat_sim_i = {} 
                for key in f.keys(): 
                    subcat_sim_i[key] = f[key]
                
                isSF = np.where(subcat_sim_i['galtype'] == 'sf') # only SF galaxies 

                sig_ms_i = smhmr.sigma_logMstar(
                        subcat_sim_i['halo.m'][isSF], subcat_sim_i['m.star'][isSF], 
                        weights=subcat_sim_i['weights'][isSF], Mhalo=Mhalo, dmhalo=dMhalo)
                sig_Mss.append(sig_ms_i)  

            sig_ms_low, sig_ms_med, sig_ms_high, sig_ms_lowlow, sig_ms_hihi = np.percentile(np.array(sig_Mss), [16, 50, 84, 2.5, 97.5])
            sigma_grid[i,j] = sig_ms_med
            sigma_grid_low[i,j] = sig_ms_low
            sigma_grid_high[i,j] = sig_ms_high

    X, Y = np.meshgrid(tduties, r_abias) 

    fig = plt.figure(figsize=(6,6)) 
    sub = fig.add_subplot(111)
    origin = 'lower'
    _CS = sub.contourf(X, Y, sigma_grid.T, origin=origin, cmap='viridis_r')
    CS = sub.contour(X, Y, sigma_grid.T, levels=_CS.levels, colors=('k',), linewidths=(0,), origin=origin) 
    manual_locations = [(0.4, 0.9), (0.45, 0.7), (0.55, 0.5), (0.9, 0.5), (2., 0.5), (3.5, 0.5), 
            (5.5, 0.5), (9, 0.5)]
    sub.clabel(CS, colors='k', fontsize=15, manual=manual_locations) #inline=1, 
    sub.plot([0.1, 10.], [0.63, 0.63], c='k', ls='--', linewidth=2, label='Tinker+(2018)') 
    sub.plot([0.1, 10.], [0.58, 0.58], c='k', ls=':', linewidth=2, label='Behroozi+(2019)') 
    #sub.fill_between([0.5, 10.], [0.57, 0.57], [0.69, 0.69], color='k', alpha=0.15, linewidth=0)
    #sub.plot([0.5, 10.], [0.57, 0.57], c='k', ls='--', linewidth=1) 
    #sub.plot([0.5, 10.], [0.69, 0.69], c='k', ls='--', linewidth=1) 
    #sub.fill_between([0.5, 10.], [0.57, 0.57], [0.69, 0.69],
    #        facecolor='none', hatch='X', edgecolor='k', linewidth=0.5)
    #sub.text(0.75, 0.63, 'Tinker+(2018)', ha='center', va='bottom', transform=sub.transAxes, fontsize=15)
    #sub.text(0.75, 0.6, 'Behroozi+(2018)', ha='center', va='top', transform=sub.transAxes, fontsize=15)
    sub.legend(loc='lower left', frameon=True, fontsize=18) 
    sub.set_xlabel(r'$t_\mathrm{duty}$ [Gyr]', fontsize=20)  
    sub.set_xlim([0.1, 10.]) 
    sub.set_xscale("log") 
    sub.set_ylabel(r'$r$ correlation coefficient', fontsize=20)  
    sub.set_ylim([0., 0.99]) 
    sub.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 0.99]) 
    sub.set_yticklabels([0., 0.2, 0.4, 0.6, 0.8, 1.]) 
    fig.savefig(''.join([UT.tex_dir(), 'figs/SHMRscatter_tduty_abias_contour.pdf']), 
            bbox_inches='tight', dpi=150) 
    plt.close()
    return None 


def _lAbramson2016SHMR(nsnap0=15, n_boot=20): 
    '''
    '''
    cat = Cat.CentralSubhalos(sigma_smhm=0.2, smf_source='li-march', nsnap0=15) 
    sh = cat.Read()
    
    # assign SFH to halos based on their SHAM M* at snapshot 15
    i_la2016 = LA2016.SHassignSFH(sh, nsnap=nsnap0, logMs_range=[9.,12.])
    hasmatch = (i_la2016 != -999) 

    # read in Louis's catalog
    cat = LA2016.catalogAbramson2016() 
    i_z0 = np.argmin(np.abs(cat['REDSHIFT'] - UT.z_nsnap(nsnap0))) #  index that best match to snapshot 
    mstar0 = np.log10(cat['MSTEL_T'][:,i_z0]) # M* @ z = z(nsnap0)  
    
    i_zf = np.argmin(np.abs(cat['REDSHIFT'] - UT.z_nsnap(1))) #  index that best match to snapshot 
    sfrf = np.log10(cat['SFR_T'][:,i_zf]) # SFR @ z = z(nsnap0)  
    mstarf = np.log10(cat['MSTEL_T'][:,i_zf]) # M* @ z = 0.05
    smhmr = Obvs.Smhmr()
    
    mh = sh['halo.m'][hasmatch]
    ms = mstarf[i_la2016[hasmatch]]
    sfr = sfrf[i_la2016[hasmatch]]

    sfcut = (sfr - ms > -11.) 
    print('%f out of %f above SF cut' % (np.sum(sfcut), len(mh))) 

    # sigma_logM* comparison 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    mbin = np.arange(12.3, 14., 0.1) 
    #m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(mh[sfcut], ms[sfcut], m_bin=mbin)
    #sub.plot(m_mid, sig_logMs, c='C1', label='SF cut')#, label='$z='+str(round(UT.z_nsnap(1),2))+'$') 

    m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(sh['halo.m.snap'+str(nsnap0)][hasmatch], 
            mstar0[i_la2016[hasmatch]], m_bin=mbin)
    sub.plot(m_mid, sig_logMs, c='k', label='$z='+str(round(UT.z_nsnap(nsnap0),2))+'$') 

    #m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(sh['halo.m'][hasmatch], mstarf[i_la2016[hasmatch]], m_bin=mbin)
    m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(mh[sfcut], ms[sfcut], m_bin=mbin)
    #sub.plot(m_mid, sig_logMs, c='C1', label='SF cut')#, label='$z='+str(round(UT.z_nsnap(1),2))+'$') 
    i_sfcut = np.where(sfcut)[0]

    siglogMs_boots = np.zeros((n_boot, len(sig_logMs)))
    for ib in range(n_boot): 
        #iboot = np.random.choice(i_hasmatch, size=np.sum(hasmatch), replace=True) 
        #_, _, sig_logMs_i, _ = smhmr.Calculate(sh['halo.m'][iboot], mstarf[i_la2016[iboot]], m_bin=mbin)
        iboot = np.random.choice(i_sfcut, size=np.sum(sfcut), replace=True) 
        _, _, sig_logMs_i, _ = smhmr.Calculate(mh[iboot], ms[iboot], m_bin=mbin)
        siglogMs_boots[ib,:] = sig_logMs_i
    sig_siglogMs = np.std(siglogMs_boots, axis=0)
    for imh in range(len(m_mid)): 
        print('at log Mh = %f sigma_Ms = %f pm %f' % (m_mid[imh], sig_logMs[imh], sig_siglogMs[imh]))
    sub.errorbar(m_mid, sig_logMs, sig_siglogMs, label='$z='+str(round(UT.z_nsnap(1),2))+'$')

    sub.set_xlabel('log $M_h$', fontsize=25) 
    sub.set_xlim([11., 14.]) 
    sub.set_ylabel('$\sigma_{\log\,M_*}$', fontsize=25) 
    sub.set_ylim([0., 0.5]) 
    sub.legend(loc='upper left', fontsize=20)
    fig.savefig(''.join([UT.tex_dir(), 'figs/_lAbramson2016SHMR.pdf']), 
            bbox_inches='tight', dpi=150) 
    plt.close()
    return None 


def _lAbramson2016SHMR_assignHalo(nsnap0=15, n_boot=20): 
    ''' Assign halos to Abramson's SFHs indirectly using abundance matching 
    then measure sigma_logM*. 
    '''
    np.random.RandomState(0)
    # read in subhalo catalog
    cat = Cat.CentralSubhalos(sigma_smhm=0.2, smf_source='li-march', nsnap0=15) 
    sh = cat.Read()
    # read in Louis's catalog
    cat = LA2016.catalogAbramson2016() 
    i_z0 = np.argmin(np.abs(cat['REDSHIFT'] - UT.z_nsnap(nsnap0))) #  index that best match to snapshot 
    mstar0 = np.log10(cat['MSTEL_T'][:,i_z0]) # M* @ z = z(nsnap0)  
    ws = LA2016.matchSMF(mstar0, UT.z_nsnap(nsnap0), logM_range=[9., 12.]) # get the weights to match SMF
    
    i_zf = np.argmin(np.abs(cat['REDSHIFT'] - UT.z_nsnap(1))) #  index that best match to snapshot 
    sfrf = np.log10(cat['SFR_T'][:,i_zf]) # SFR @ z = z(nsnap0)  
    mstarf = np.log10(cat['MSTEL_T'][:,i_zf]) # M* @ z = 0.05

    # assign halo to the SFHs
    mhalo0, mmax0, i_halo = LA2016.assignHalo(mstar0, nsnap0, sh, logM_range=[9., 12.], dlogMs=0.2) 

    mhalof = sh['halo.m'][i_halo]
    mmaxf = sh['m.max'][i_halo]

    sfcut = (sfrf - mstarf > -11.) 
    print('%f out of %f above SF cut' % (np.sum(sfcut), len(mstarf))) 
    smhmr = Obvs.Smhmr()

    # sigma_logM* comparison 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    mbin = np.arange(12.3, 14., 0.1) 
    m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(mhalo0, mstar0, weights=ws, m_bin=mbin)
    sub.plot(m_mid, sig_logMs, c='k', label='$z='+str(round(UT.z_nsnap(nsnap0),2))+'$') 

    #m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(mhalof, mstarf, weights=ws, m_bin=mbin)
    m_mid, mu_logMs, sig_logMs, _ = smhmr.Calculate(mhalof[sfcut], mstarf[sfcut], weights=ws[sfcut], m_bin=mbin)
    #sub.plot(m_mid, sig_logMs, c='C1', label='SF cut')#, label='$z='+str(round(UT.z_nsnap(1),2))+'$') 
    i_sfcut = np.where(sfcut)[0]

    siglogMs_boots = np.zeros((n_boot, len(sig_logMs)))
    for ib in range(n_boot): 
        _, _, i_halo = LA2016.assignHalo(mstar0, nsnap0, sh, logM_range=[9., 12.], dlogMs=0.2) 
        mhalof = sh['halo.m'][i_halo]
        mmaxf = sh['m.max'][i_halo]

        iboot = np.random.choice(i_sfcut, size=np.sum(sfcut), replace=True) 
        _, _, sig_logMs_i, _ = smhmr.Calculate(mhalof[iboot], mstarf[iboot], weights=ws[iboot], m_bin=mbin)
        siglogMs_boots[ib,:] = sig_logMs_i

    sig_siglogMs = np.zeros(len(sig_logMs))
    for i in range(len(sig_logMs)):
        nonzero = (siglogMs_boots[:,i] > 0.)
        sig_siglogMs[i] = np.std(siglogMs_boots[:,i][nonzero])

    sub.errorbar(m_mid, sig_logMs, sig_siglogMs, label='$z='+str(round(UT.z_nsnap(1),2))+'$')

    f_sigma = open(''.join([UT.tex_dir(), 'dat/', 'Abramson_sigma_logMs_estimates.txt']), 'w') 
    f_sigma.write("# sigma_logM* for Abramson et al. (2016)'s ~2000 SFHs \n")
    f_sigma.write("# the SFHs are connected to halos using abundance matching \n") 
    f_sigma.write("# log M*: sigma_logM* \pm sigma_sigma_logM* \n") 
    for imh in range(len(m_mid)): 
        print('at log Mh = %f sigma_Ms = %f pm %f' % (m_mid[imh], sig_logMs[imh], sig_siglogMs[imh]))
        f_sigma.write('%f: %f +/- %f \n' % (m_mid[imh], sig_logMs[imh], sig_siglogMs[imh]))
    f_sigma.close() 
    sub.set_xlabel('log $M_h$', fontsize=25) 
    sub.set_xlim([11., 14.]) 
    sub.set_ylabel('$\sigma_{\log\,M_*}$', fontsize=25) 
    sub.set_ylim([0., 0.5]) 
    sub.legend(loc='upper left', fontsize=20)
    fig.savefig(''.join([UT.tex_dir(), 'figs/_lAbramson2016SHMR_assignHalo.pdf']), 
            bbox_inches='tight', dpi=150) 
    plt.close()
    return None 


def SHMRscatter_tduty_abias_z1sigma(Mhalo=12, dMhalo=0.5):
    ''' Scatter in the Stellar to Halo Mass Relation (i.e.  sigma_logM*(M_h = 10^12)
    as a function of duty cycle timescale (t_duty) for different z=1 simga_logM* initial
    conditions.
    '''
    # calculate the scatters from the ABC posteriors 
    smhmr = Obvs.Smhmr()
    tscales = ['0.5', '1', '2', '5', '10']
    tduties = [0.5, 1., 2., 5., 9.75]  #hardcoded
    iters = [14, 14, 14, 14, 14] # iterations of ABC
    nparticles = [1000, 1000, 1000, 1000, 1000]

    fig = plt.figure(figsize=(10,5)) 
    sub1 = fig.add_subplot(121)
    sub2 = fig.add_subplot(122) 
    for siglogM in [0.35, 0.45]:
        for i_a, abias in enumerate([0., 0.5, 0.99]): 
            if abias > 0.:
                runs = ['rSFH_abias%s_%sgyr.sfsmf.sigma_z1_%.2f.sfsbroken' % (str(abias), tt, siglogM) for tt in tscales]
            else: 
                runs = ['randomSFH%sgyr.sfsmf.sigma_z1_%.2f.sfsbroken' % (tt, siglogM) for tt in tscales]

            sigMs = np.zeros((5, len(tduties)))
            for i_t, tduty in enumerate(tduties): 
                abc_dir = os.path.join(UT.dat_dir(), 'abc', runs[i_t], 'model') # ABC directory 
                sig_Mss = [] 
                for i in range(10): 
                    fabc = os.path.join(abc_dir, 'model.theta%i.t%i.p' % (i, iters[i_t]))

                    f = pickle.load(open(fabc, 'rb'))
                    subcat_sim_i = {} 
                    for key in f.keys(): 
                        subcat_sim_i[key] = f[key]
                    
                    isSF = np.where(subcat_sim_i['galtype'] == 'sf') # only SF galaxies 

                    sig_ms_i = smhmr.sigma_logMstar(
                            subcat_sim_i['halo.m'][isSF], subcat_sim_i['m.star'][isSF], 
                            weights=subcat_sim_i['weights'][isSF], Mhalo=Mhalo, dmhalo=dMhalo)
                    sig_Mss.append(sig_ms_i)  

                sig_ms_low, sig_ms_med, sig_ms_high, sig_ms_lowlow, sig_ms_hihi = np.percentile(np.array(sig_Mss), [16, 50, 84, 2.5, 97.5])
                sigMs[0, i_t] = sig_ms_med
                sigMs[1, i_t] = sig_ms_low
                sigMs[2, i_t] = sig_ms_high
                sigMs[3, i_t] = sig_ms_lowlow
                sigMs[4, i_t] = sig_ms_hihi

            if siglogM == 0.35: 
                sub1.fill_between(tduties, sigMs[1,:], sigMs[2,:], color='C'+str(i_a), alpha=0.5, linewidth=0) 
                sim_plt = sub1.fill_between([0.1, 11.], [0.3, 0.3], [0.36, 0.36], facecolor='none', hatch='...', edgecolor='k', linewidth=0.)
            elif siglogM == 0.45: 
                sub2.fill_between(tduties, sigMs[1,:], sigMs[2,:], color='C'+str(i_a), alpha=0.5, linewidth=0) 
                sub2.fill_between([0.1, 11.], [0.3, 0.3], [0.36, 0.36], facecolor='none', hatch='...', edgecolor='k', linewidth=0.)
            print('sigma_logM*=%.2f; r=%.2f' % (siglogM, abias))  
            print('tduty=0.5, sigma_logM*=%f +/- %f, %f' % (sigMs[0,0], sigMs[2,0]-sigMs[0,0], sigMs[0,0]-sigMs[1,0]))
            print('tduty=10, sigma_logM*=%f +/- %f, %f' % (sigMs[0,-1], sigMs[2,-1]-sigMs[0,-1], sigMs[0,-1]-sigMs[1,-1])) 
    sub1.text(0.95, 0.05, r'$\sigma_{\log M_*}(z\sim1) = 0.35$ dex', ha='right', va='bottom', 
            transform=sub1.transAxes, fontsize=20)
    sub1.set_xlim(0.5, 10.) # x-axis
    sub1.set_xscale('log') 
    sub1.set_ylabel(r'$\sigma_{\log M_*} \Big(M_{h} = 10^{'+str(Mhalo)+r'} M_\odot \Big)$', fontsize=25) # y-axis
    sub1.set_ylim(0.1, 0.4) 
    sub1.set_yticks([0.1, 0.2, 0.3, 0.4])#, 0.6]) 
    sub1.legend([sim_plt], ['Cao+(2019)'], loc=(0.425, 0.15), handletextpad=0.4, prop={'size': 18})
    
    # right panel: x,y axis
    sub2.set_xlim(0.5, 10.) 
    sub2.set_xscale('log') 
    sub2.set_ylim(0.1, 0.4) 
    sub2.set_yticks([0.1, 0.2, 0.3, 0.4])
    # right panel: legends  
    abc_post1 = sub2.fill_between([0], [0], [0.1], color='C0', label='$r=0$')
    abc_post2 = sub2.fill_between([0], [0], [0.1], color='C1', label='$r=0.5$') 
    abc_post3 = sub2.fill_between([0], [0], [0.1], color='C2', label='$r=0.99$') 
    leg2 = sub2.legend(loc=(0.52, 0.15), handletextpad=0.4, frameon=False, fontsize=18) 
   
    sub2.text(0.95, 0.05, r'$\sigma_{\log M_*}(z\sim1) = 0.45$ dex', ha='right', va='bottom', 
            transform=sub2.transAxes, fontsize=20)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('$t_\mathrm{duty}$ [Gyr]', labelpad=10, fontsize=22) 
    fig.subplots_adjust(wspace=0.2)
    fig.savefig(''.join([UT.tex_dir(), 'figs/SHMRscatter_tduty_abias_z1sigma.pdf']), 
            bbox_inches='tight', dpi=150) 
    plt.close()
    return None 


def siglogM_zevo(Mhalo=12, dMhalo=0.5):
    ''' redshift evolution of the sigma_logM*(M_h = 10^12), the scatter in the Stellar to
    Halo Mass Relation,  
    '''
    # calculate the scatters from the ABC posteriors 
    smhmr = Obvs.Smhmr()

    fig = plt.figure(figsize=(15,5)) 
    for i_sig, siglogM in enumerate([0.2, 0.35, 0.45]):
        sub = fig.add_subplot(1,3,i_sig+1)
        for tt in ['0.1']:  
            for i_a, abias in enumerate([0., 0.5, 0.99]): 
                if abias > 0.:
                    if siglogM == 0.2: run = 'rSFH_abias%s_%sgyr.sfsmf.sfsbroken' % (str(abias), tt) 
                    else: run = 'rSFH_abias%s_%sgyr.sfsmf.sigma_z1_%.2f.sfsbroken' % (str(abias), tt, siglogM)
                else: 
                    if siglogM == 0.2: run = 'randomSFH%sgyr.sfsmf.sfsbroken' % tt
                    else: run = 'randomSFH%sgyr.sfsmf.sigma_z1_%.2f.sfsbroken' % (tt, siglogM) 

                abc_dir = os.path.join(UT.dat_dir(), 'abc', run, 'model') # ABC directory 
                sig_Mss = [] 
                for i in range(100): 
                    fabc = os.path.join(abc_dir, 'model.theta%i.t%i.p' % (i, 14))

                    f = pickle.load(open(fabc, 'rb'))
                    subcat_sim_i = {} 
                    for key in f.keys(): 
                        subcat_sim_i[key] = f[key]
                    
                    isSF = np.where(subcat_sim_i['galtype'] == 'sf') # only SF galaxies 

                    sig_ms_i = smhmr.sigma_logMstar(
                            subcat_sim_i['halo.m'][isSF], subcat_sim_i['m.star'][isSF], 
                            weights=subcat_sim_i['weights'][isSF], Mhalo=Mhalo, dmhalo=dMhalo)
                    sig_Mss.append(sig_ms_i)  
                sig_ms_low, sig_ms_med, sig_ms_high, sig_ms_lowlow, sig_ms_hihi = np.percentile(np.array(sig_Mss), [16, 50, 84, 2.5, 97.5])
                sub.fill_between([1., 0.], [siglogM, sig_ms_low], [siglogM, sig_ms_high], color='C'+str(i_a), alpha=1, linewidth=0)
                print('sigma_M*|Mh(z=1) = %.2f, r = %.1f --- sigma_M*|Mh = %f' % (siglogM, abias, sig_ms_med))
        sub.plot([1., 0.], [siglogM, siglogM], ls='--', c='k') 
        sub.set_xlim(1., 0.) # x-axis
        sub.set_ylim(0.2, 0.5) 
        sub.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5]) 
        if i_sig == 0: 
            sub.set_ylabel(r'$\sigma_{\log M_*} \Big(M_{h} = 10^{'+str(Mhalo)+r'} M_\odot \Big)$', fontsize=25) # y-axis
            sub.text(0.05, 0.95, r'$t_{\rm duty} = 0.1$ Gyr', ha='left', va='top', transform=sub.transAxes, fontsize=25)
        elif i_sig == 2: # right panel: legends  
            abc_post1 = sub.fill_between([0], [0], [0.1], color='C0', label='$r=0$')
            abc_post2 = sub.fill_between([0], [0], [0.1], color='C1', label='$r=0.5$') 
            abc_post3 = sub.fill_between([0], [0], [0.1], color='C2', label='$r=0.99$') 
            leg2 = sub.legend(loc=(0.52, 0.025), handletextpad=0.4, frameon=False, fontsize=18) 
        sub.set_xlabel('Redshift ($z$)', fontsize=22) 
    fig.subplots_adjust(wspace=0.2)
    fig.savefig(''.join([UT.tex_dir(), 'figs/siglogM_zevo.pdf']), bbox_inches='tight', dpi=150) 
    plt.close()
    return None 


def _SHMRscatter_tduty_SFSflexVSanchored(Mhalo=12, dMhalo=0.5, Mstar=10.5, dMstar=0.5):
    ''' Comparison of the SHMR(t_duty) for flexible SFS versus anchored SFS prescriptions.
    '''
    tduties = [0.5, 1., 2., 5., 7.47]  #hardcoded
    iters = [14 for i in range(len(tduties))] # 13, 13, 12, 12, 12] # iterations of ABC

    smhmr = Obvs.Smhmr()
    # calculate the scatters from the ABC posteriors 
    for sfs in ['flex', 'anchored']:  
        runs = ['randomSFH'+str(t)+'gyr.sfs'+sfs for t in [0.5, 1, 2, 5, 10]] 
        sigMs = np.zeros((3, len(tduties)))
        sigMh = np.zeros((3, len(tduties)))
        for i_t, tduty in enumerate(tduties): 
            abc_dir = ''.join([UT.dat_dir(), 'abc/', runs[i_t], '/model/']) # ABC directory 
            sig_Mss, sig_Mhs = [], [] 
            for i in range(100): 
                #f = h5py.File(''.join([abc_dir, 'model.theta', str(i), '.t', str(iters[i_t]), '.hdf5']), 'r') 
                #subcat_sim_i = {} 
                #for key in f.keys(): 
                #    subcat_sim_i[key] = f[key].value
                f = pickle.load(open(''.join([abc_dir, 'model.theta', str(i), '.t', str(iters[i_t]), '.p']), 'rb'))
                subcat_sim_i = {} 
                for key in f.keys(): 
                    subcat_sim_i[key] = f[key]
                
                isSF = np.where(subcat_sim_i['galtype'] == 'sf') # only SF galaxies 

                sig_ms_i = smhmr.sigma_logMstar(
                        subcat_sim_i['halo.m'][isSF], subcat_sim_i['m.star'][isSF], 
                        weights=subcat_sim_i['weights'][isSF], Mhalo=Mhalo, dmhalo=dMhalo)
                sig_mh_i = smhmr.sigma_logMhalo(
                        subcat_sim_i['halo.m'][isSF], subcat_sim_i['m.star'][isSF], 
                        weights=subcat_sim_i['weights'][isSF], Mstar=Mstar, dmstar=dMstar)
                sig_Mss.append(sig_ms_i)  
                sig_Mhs.append(sig_mh_i) 

            sig_ms_low, sig_ms_med, sig_ms_high = np.percentile(np.array(sig_Mss), [16, 50, 84]) 
            sig_mh_low, sig_mh_med, sig_mh_high = np.percentile(np.array(sig_Mhs), [16, 50, 84]) 

            sigMs[0, i_t] = sig_ms_med
            sigMs[1, i_t] = sig_ms_low
            sigMs[2, i_t] = sig_ms_high
            
            sigMh[0, i_t] = sig_mh_med
            sigMh[1, i_t] = sig_mh_low
            sigMh[2, i_t] = sig_mh_high
        if sfs == 'flex': 
            sigMs_flex = sigMs
            sigMh_flex = sigMh
        elif sfs == 'anchored': 
            sigMs_anch = sigMs
            sigMh_anch = sigMh
            
    # make figure 
    fig = plt.figure(figsize=(10,5)) 
    bkgd = fig.add_subplot(111, frameon=False)
    sub = fig.add_subplot(121)
    # ABC posteriors 
    sub.errorbar(tduties, sigMs_flex[0,:], yerr=[sigMs_flex[0,:]-sigMs_flex[1,:], sigMs_flex[2,:]-sigMs_flex[0,:]], 
            fmt='.k', label='Flex SFS') 
    sub.errorbar(tduties, sigMs_anch[0,:], yerr=[sigMs_anch[0,:]-sigMs_anch[1,:], sigMs_anch[2,:]-sigMs_anch[0,:]], 
            fmt='.C1', label='Anchored SFS') 
    sub.legend(loc='lower right', fontsize=20) 
    sub.set_xlim([0.45, 10.]) # x-axis
    sub.set_xscale('log') 
    sub.set_ylabel(r'$\sigma_{M_*} \Big(M_\mathrm{halo} = 10^{'+str(Mhalo)+r'} M_\odot \Big)$', fontsize=20) # y-axis
    sub.set_ylim([0., 0.6]) 
    sub.set_yticks([0., 0.2, 0.4, 0.6]) 
    
    sub = fig.add_subplot(122) 
    # ABC posteriors 
    sub.errorbar(tduties, sigMh_flex[0,:], yerr=[sigMh_flex[0,:]-sigMh_flex[1,:], sigMh_flex[2,:]-sigMh_flex[0,:]], 
            fmt='.k') 
    sub.errorbar(tduties, sigMh_anch[0,:], yerr=[sigMh_anch[0,:]-sigMh_anch[1,:], sigMh_anch[2,:]-sigMh_anch[0,:]], fmt='.C1') 
    sub.set_xlim([0.45, 10.]) # x-axis
    sub.set_xscale('log') 
    sub.set_ylabel(r'$\sigma_{M_\mathrm{halo}} \Big(M_* = 10^{'+str(Mstar)+r'} M_\odot \Big)$', fontsize=20) # y-axis
    sub.set_ylim([0., 0.75]) 
    sub.set_yticks([0., 0.2, 0.4, 0.6]) 
    
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('$t_\mathrm{duty}$ [Gyr]', labelpad=10, fontsize=22) 
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(''.join([UT.tex_dir(), 'figs/_SHMRscatter_tduty_SFSflexVSanchored.pdf']), 
            bbox_inches='tight', dpi=150) 
    plt.close()
    return None 


def _SHMRscatter_tduty_narrowSFS(Mhalo=12, dMhalo=0.5, Mstar=10.5, dMstar=0.5):
    ''' Comparison of the SHMR(t_duty) for SFS with 0.3 dex scatter 
    versus SFS with 0.2 dex scatter.
    '''
    tduties = [0.5, 1., 2., 5., 10.]  #hardcoded
    iters = [14 for i in range(len(tduties))] # 13, 13, 12, 12, 12] # iterations of ABC

    smhmr = Obvs.Smhmr()
    # calculate the scatters from the ABC posteriors 
    for prefix in ['randomSFH', 'rSFH_0.2sfs_']:  
        runs = [prefix+str(t)+'gyr.sfsflex' for t in [0.5, 1, 2, 5, 10]] 
        sigMs = np.zeros((3, len(tduties)))
        sigMh = np.zeros((3, len(tduties)))
        for i_t, tduty in enumerate(tduties): 
            abc_dir = ''.join([UT.dat_dir(), 'abc/', runs[i_t], '/model/']) # ABC directory 
            sig_Mss, sig_Mhs = [], [] 
            for i in range(100): 
                #f = h5py.File(''.join([abc_dir, 'model.theta', str(i), '.t', str(iters[i_t]), '.hdf5']), 'r') 
                #subcat_sim_i = {} 
                #for key in f.keys(): 
                #    subcat_sim_i[key] = f[key].value
                f = pickle.load(open(''.join([abc_dir, 'model.theta', str(i), '.t', str(iters[i_t]), '.p']), 'rb'))
                subcat_sim_i = {} 
                for key in f.keys(): 
                    subcat_sim_i[key] = f[key]
                
                isSF = np.where(subcat_sim_i['galtype'] == 'sf') # only SF galaxies 

                sig_ms_i = smhmr.sigma_logMstar(
                        subcat_sim_i['halo.m'][isSF], subcat_sim_i['m.star'][isSF], 
                        weights=subcat_sim_i['weights'][isSF], Mhalo=Mhalo, dmhalo=dMhalo)
                sig_mh_i = smhmr.sigma_logMhalo(
                        subcat_sim_i['halo.m'][isSF], subcat_sim_i['m.star'][isSF], 
                        weights=subcat_sim_i['weights'][isSF], Mstar=Mstar, dmstar=dMstar)
                sig_Mss.append(sig_ms_i)  
                sig_Mhs.append(sig_mh_i) 

            sig_ms_low, sig_ms_med, sig_ms_high = np.percentile(np.array(sig_Mss), [16, 50, 84]) 
            sig_mh_low, sig_mh_med, sig_mh_high = np.percentile(np.array(sig_Mhs), [16, 50, 84]) 

            sigMs[0, i_t] = sig_ms_med
            sigMs[1, i_t] = sig_ms_low
            sigMs[2, i_t] = sig_ms_high
            
            sigMh[0, i_t] = sig_mh_med
            sigMh[1, i_t] = sig_mh_low
            sigMh[2, i_t] = sig_mh_high
        if prefix == 'randomSFH':  
            sigMs0 = sigMs
            sigMh0 = sigMh
        elif prefix == 'rSFH_0.2sfs_':
            sigMs_narr = sigMs
            sigMh_narr = sigMh
            
    # make figure 
    fig = plt.figure(figsize=(10,5)) 
    bkgd = fig.add_subplot(111, frameon=False)
    sub = fig.add_subplot(121)
    # ABC posteriors 
    sub.errorbar(tduties, sigMs0[0,:], yerr=[sigMs0[0,:]-sigMs0[1,:], sigMs0[2,:]-sigMs0[0,:]], fmt='.k', 
            label=r'$\sigma_\mathrm{SFS} = 0.3$ dex') 
    sub.errorbar(tduties, sigMs_narr[0,:], yerr=[sigMs_narr[0,:]-sigMs_narr[1,:], sigMs_narr[2,:]-sigMs_narr[0,:]], 
            fmt='.C1', label=r'$\sigma_\mathrm{SFS} = 0.2$ dex') 
    sub.legend(loc='lower right', fontsize=20) 
    sub.set_xlim([0.45, 10.]) # x-axis
    sub.set_xscale('log') 
    sub.set_ylabel(r'$\sigma_{M_*} \Big(M_\mathrm{halo} = 10^{'+str(Mhalo)+r'} M_\odot \Big)$', fontsize=20) # y-axis
    sub.set_ylim([0., 0.6]) 
    sub.set_yticks([0., 0.2, 0.4, 0.6]) 
    
    sub = fig.add_subplot(122) 
    # ABC posteriors 
    sub.errorbar(tduties, sigMh0[0,:], yerr=[sigMh0[0,:]-sigMh0[1,:], sigMh0[2,:]-sigMh0[0,:]], fmt='.k') 
    sub.errorbar(tduties, sigMh_narr[0,:], yerr=[sigMh_narr[0,:]-sigMh_narr[1,:], sigMh_narr[2,:]-sigMh_narr[0,:]], fmt='.C1') 
    sub.set_xlim([0.45, 10.]) # x-axis
    sub.set_xscale('log') 
    sub.set_ylabel(r'$\sigma_{M_\mathrm{halo}} \Big(M_* = 10^{'+str(Mstar)+r'} M_\odot \Big)$', fontsize=20) # y-axis
    sub.set_ylim([0., 0.75]) 
    sub.set_yticks([0., 0.2, 0.4, 0.6]) 
    
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('$t_\mathrm{duty}$ [Gyr]', labelpad=10, fontsize=22) 
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(''.join([UT.tex_dir(), 'figs/_SHMRscatter_tduty_narrowSFS.pdf']), 
            bbox_inches='tight', dpi=150) 
    plt.close()
    return None 


def _convert_sigma_lit(name): 
    ''' convert whatever measurement from the literature (specified by name) 
    to sigma_M* or sigma_Mh. 
    '''
    print('%s' % name) 
    if name == 'conroy2007': 
        # vdisp and halo mass in stellar mass bins 
        h = 0.7 
        Ms_bin = np.array([9.6, 10.4]) - np.log10(h**2) 
        Ms_mid = 10.2 - np.log10(h**2) 
        print('constraint for M* bin: %f < M* < %f' % (Ms_bin[0], Ms_bin[1])) 

        M200overMs = np.array([74, 105, 132]) 
        M200 = Ms_mid + np.log10(M200overMs) - np.log10(h)
        print('M200-1sigma, M200, M200+1sigma = %f, %f, %f' % (M200[0], M200[1], M200[2]))
        print('sigma_Mh = %f' % np.max([M200[1] - M200[0], M200[2] - M200[1]])) 
    elif name == 'han2015': 
        h = 0.7 
        Ms = np.log10(1.4e10) - np.log10(h**2) 
        Mh = np.array([11.14, 11.713, 12.31]) - np.log10(h) 
        print('constraint for M* = %f' % Ms) 
        print('Mh-1sigma, Mh, Mh+1sigma = %f, %f, %f' % tuple(Mh))
        print('sigma_Mh = %f' % np.max([Mh[1] - Mh[0], Mh[2] - Mh[1]])) 
    return None 

"""
    def sigMstar_tduty(Mhalo=12, dMhalo=0.5):
        ''' Figure plotting sigmaMstar at M_halo = Mhalo for different 
        duty cycle time (t_duty). 
        '''
        runs = ['randomSFH_0.5gyr', 'randomSFH_1gyr', 'randomSFH_2gyr', 'randomSFH_5gyr', 'test0']
        tduties = [0.5, 1., 2., 5., 7.47]  #hardcoded
        iters = [13, 14, 14, 14, 14] # iterations of ABC
        nparticles = [1000, 1000, 1000, 1000, 1000]
        
        smhmr = Obvs.Smhmr()
        
        sigMstar_med, sigMstar_low, sigMstar_high = [], [], [] 
        for i_t, tduty in enumerate(tduties): 
            abc_dir = UT.dat_dir()+'abc/'+runs[i_t]+'/model/' # ABC directory 
            # theta median
            #sig_mstar_meds = np.zeros(10) 
            #for i in range(10): 
            #    f = h5py.File(''.join([abc_dir, 'model.theta_median', str(i), '.t', str(iters[i_t]), '.hdf5']), 'r') 
            #    subcat_sim = {} 
            #    for key in f.keys(): 
            #        subcat_sim[key] = f[key].value
            #    isSF = np.where(subcat_sim['gclass'] == 'sf') # only SF galaxies 

            #    sig_mstar_meds[i] = smhmr.sigma_logMstar(
            #            subcat_sim['halo.m'][isSF], subcat_sim['m.star'][isSF], 
            #            weights=subcat_sim['weights'][isSF], Mhalo=Mhalo, dmhalo=dMhalo)
            #sig_mstar_med = np.average(sig_mstar_meds)

            # other thetas  
            sig_mstars = [] 
            for i in range(200): 
                f = h5py.File(''.join([abc_dir, 'model.theta', str(i), '.t', str(iters[i_t]), '.hdf5']), 'r') 
                subcat_sim_i = {} 
                for key in f.keys(): 
                    subcat_sim_i[key] = f[key].value
                
                isSF = np.where(subcat_sim_i['gclass'] == 'sf') # only SF galaxies 

                sig_mstar_i = smhmr.sigma_logMstar(
                        subcat_sim_i['halo.m'][isSF], subcat_sim_i['m.star'][isSF], 
                        weights=subcat_sim_i['weights'][isSF], Mhalo=Mhalo, dmhalo=dMhalo)
                sig_mstars.append(sig_mstar_i)  

            sig_mstar_low, sig_mstar_med, sig_mstar_high = np.percentile(np.array(sig_mstars), [16, 50, 84]) 
            sigMstar_med.append(sig_mstar_med)
            sigMstar_low.append(sig_mstar_low) 
            sigMstar_high.append(sig_mstar_high) 
        sigMstar_med = np.array(sigMstar_med) 
        sigMstar_low = np.array(sigMstar_low) 
        sigMstar_high = np.array(sigMstar_high) 
        
        pretty_colors = prettycolors() 
        # make figure 
        fig = plt.figure(figsize=(5,5)) 
        sub = fig.add_subplot(111)

        # plot constraints from literature
        # Tinker et al. (2013) for star-forming galaxies 0.2 < z < 0.48 (COSMOS)
        tinker2013 = sub.fill_between([0., 10.], [0.15, 0.15], [0.27, 0.27], alpha=0.1, label='Tinker+(2013)', color=pretty_colors[7], linewidth=0) 
        # More et al. (2011) SMHMR of starforming centrals (SDSS)
        more2011 = sub.fill_between([0., 10.], [0.07, 0.07], [0.26, 0.26], label='More+(2011)', facecolor="none", hatch='/', edgecolor='k', linewidth=0.5)
        # Leauthaud et al. (2012) all galaxies 0.2 < z < 0.48 (COSMOS)
        leauthaud2012 = sub.fill_between([0., 10.], [0.191-0.031, 0.191-0.031], [0.191+0.031, 0.191+0.031], alpha=0.2, label='Leauthaud+(2012)', color=pretty_colors[1], linewidth=0)
        # Reddick et al. (2013) Figure 7. (constraints from conditional SMF)
        reddick2013 = sub.fill_between([0., 10.], [0.187, 0.187], [0.233, 0.233], label='Reddick+(2013)', facecolor="none", hatch='X', edgecolor='k', linewidth=0.25)
        # Zu & Mandelbaum (2015) SDSS constraints on iHOD parameters
        zu2015 = sub.fill_between([0., 10.], [0.21, 0.21], [0.24, 0.24], alpha=0.2, label='Zu+(2015)', color=pretty_colors[1], linewidth=0)
        # Meng Gu et al. (2016) 
        #gu2016, = sub.plot([0., 10.], [0.32, 0.32], ls='--', c='k') 

        abc_post = sub.errorbar(tduties, sigMstar_med, 
                yerr=[sigMstar_med-sigMstar_low, sigMstar_high-sigMstar_med], fmt='.k') 
        #sub.scatter(tduties, sigMstar_med, c='k') 

        #legend1 = sub.legend([more2011, leauthaud2012, gu2016], ['More+(2011)', 'Leauthaud+(2012)', 'Gu+(2016)'], loc='upper left', prop={'size': 15})
        legend1 = sub.legend([abc_post, more2011, leauthaud2012], ['ABC Posteriors', 'More+(2011)', 'Leauthaud+(2012)'], loc='upper left', prop={'size': 15})
        sub.legend([reddick2013, tinker2013], ['Reddick+(2013)', 'Tinker+(2013)'], loc='lower right', prop={'size': 15})
        plt.gca().add_artist(legend1)
        # x-axis
        sub.set_xlabel('$t_\mathrm{duty}$ [Gyr]', fontsize=20)
        sub.set_xlim([0., 7.8]) 
        
        # y-axis
        sub.set_ylabel(r'$\sigma_{M_*} \Big(M_\mathrm{halo} = 10^{'+str(Mhalo)+r'} M_\odot \Big)$', fontsize=20)
        sub.set_ylim([0., 0.5]) 
        
        fig.savefig(''.join([UT.tex_dir(), 'figs/sigMstar_tduty.pdf']), 
                bbox_inches='tight', dpi=150) 
        plt.close()
        return None 


    def sigMstar_tduty_fid(Mhalo=12, dMhalo=0.5):
        ''' Figure plotting sigmaMstar at M_halo = Mhalo for different 
        duty cycle time (t_duty) with fiducial SFMS parameter values rather 
        than ABC values. 
        '''
        # read in parameter values for randomSFH_1gyr
        abcout = ABC.readABC('randomSFH_1gyr', 13)
        # the median theta will be designated the fiducial parameter values 
        theta_fid = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) 
                for i in range(len(abcout['theta'][0]))]
        
        runs = ['randomSFH_0.5gyr', 'randomSFH_1gyr', 'randomSFH_2gyr', 'randomSFH_5gyr', 'randomSFH_10gyr']
        tduties = [0.5, 1., 2., 5., 10.]  #hardcoded
        
        smhmr = Obvs.Smhmr()

        sigMstar_fid = []
        for i_t, tduty in enumerate(tduties): 
            subcat_sim = ABC.model(runs[i_t], theta_fid, 
                    nsnap0=15, sigma_smhm=0.2, downsampled='14') 
            isSF = np.where(subcat_sim['gclass'] == 'sf') # only SF galaxies 

            sig_mstar_fid = smhmr.sigma_logMstar(
                    subcat_sim['halo.m'][isSF], subcat_sim['m.star'][isSF], 
                    weights=subcat_sim['weights'][isSF], Mhalo=Mhalo, dmhalo=dMhalo)

            sigMstar_fid.append(sig_mstar_fid)
        sigMstar_fid = np.array(sigMstar_fid)
        
        # make figure 
        fig = plt.figure(figsize=(5,5)) 
        sub = fig.add_subplot(111)
        sub.scatter(tduties, sigMstar_fid) 
        # x-axis
        sub.set_xlabel('$t_\mathrm{duty}$ [Gyr]', fontsize=20)
        sub.set_xlim([0., 10.]) 
        
        # y-axis
        sub.set_ylabel('$\sigma_{M_*}(M_\mathrm{halo} = 10^{'+str(Mhalo)+'} M_\odot)$', fontsize=20)
        sub.set_ylim([0., 0.5]) 
        
        fig.savefig(''.join([UT.tex_dir(), 'figs/sigMstar_tduty_fid.pdf']), 
                bbox_inches='tight', dpi=150) 
        plt.close()
        return None 
"""

if __name__=="__main__": 
    #groupcatSFMS(mrange=[10.6,10.8])
    #SFMSprior_z1()
    #sigMstar_tduty_fid(Mhalo=12, dMhalo=0.1)
    #sigMstar_tduty(Mhalo=12, dMhalo=0.1)
    #SHMRscatter_tduty(Mhalo=12, dMhalo=0.1, Mstar=10.5, dMstar=0.1)
    #SHMRscatter_tduty_abias(Mhalo=12, dMhalo=0.1, Mstar=10.5, dMstar=0.1)
    #qaplotABC(runs=['randomSFH10gyr.sfsmf.sfsbroken', 'randomSFH1gyr.sfsmf.sfsbroken'], Ts=[14, 14], dMhalo=0.1)
    #SHMRscatter_tduty(Mhalo=12, dMhalo=0.1, Mstar=10., dMstar=0.1)
    #SHMRscatter_tduty_v2(Mhalo=12, dMhalo=0.1)
    #SHMRscatter_tduty_v2(Mhalo=11.5, dMhalo=0.1)
    #SHMRscatter_tduty_abias(Mhalo=12, dMhalo=0.1, Mstar=10.5, dMstar=0.2)
    #SHMRscatter_tduty_abias_v2(Mhalo=12, dMhalo=0.1, Mstar=10.5, dMstar=0.1)
    #SHMRscatter_tduty_abias_contour(Mhalo=12, dMhalo=0.1, niter=14)
    #Mhacc_dSFR(['rSFH_abias0.5_0.5gyr.sfsmf.sfsbroken', 'rSFH_abias0.99_0.5gyr.sfsmf.sfsbroken'], 14)
    #SHMRscatter_tduty_abias_z1sigma(Mhalo=12, dMhalo=0.1)
    siglogM_zevo(Mhalo=12, dMhalo=0.1)
    #fQ_fSFMS()
    #SFHmodel()
    #Illustris_SFH()
    #_lAbramson2016SHMR(nsnap0=15, n_boot=100)
    #_lAbramson2016SHMR_assignHalo(nsnap0=15, n_boot=100)
    #_SHMRscatter_tduty_SFSflexVSanchored(Mhalo=12, dMhalo=0.1, Mstar=10.5, dMstar=0.2)
    #_SHMRscatter_tduty_narrowSFS(Mhalo=12, dMhalo=0.2, Mstar=10.5, dMstar=0.2)
    #_convert_sigma_lit('han2015')
