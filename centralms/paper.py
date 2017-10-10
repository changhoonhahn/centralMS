#!/usr/bin/env python
''' 

Make figures for paper 


'''
import numpy as np 
import corner as DFM
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal as MNorm
from letstalkaboutquench.fstarforms import fstarforms

import util as UT
import abcee as ABC
import catalog as Cat
import observables as Obvs

from ChangTools.plotting import prettycolors
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
    fSFMS = fstarforms() 
    fit_logm, fit_logsfr = fSFMS.fit(gc_cat['mass'], gc_cat['sfr'], method='gaussmix', fit_range=mrange)
    _, fit_fsfms = fSFMS.frac_SFMS()
    i_fit = np.abs(fit_logm - np.mean(mrange)).argmin()

    # log SFR - log M* highlighting where the SFMS lies 
    sub1 = fig.add_subplot(1,2,1)
    DFM.hist2d(gc_cat['mass'], gc_cat['sfr'], color='#ee6a50',
            levels=[0.68, 0.95], range=[[9., 12.], [-3.5, 1.5]], 
            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub1) 
    #sub1.vlines(mrange[0], -5., 2., color='k', linewidth=2, linestyle='--')
    #sub1.vlines(mrange[1], -5., 2., color='k', linewidth=2, linestyle='--')
    #sub1.fill_between(mrange, [2.,2.], [-5.,-5], color='#1F77B4', alpha=0.25)
    sub1.fill_between(mrange, [2.,2.], [-5.,-5], color='k', linewidth=0, alpha=0.25)
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

    sub2.set_xlim([-9.5, -13.25]) 
    sub2.set_xticks([-10., -11., -12., -13.])
    sub2.set_xlabel('log$(\; \mathrm{SSFR}\; [\mathrm{yr}^{-1}]\;)$', fontsize=20)
    sub2.set_ylim([0., 1.5]) 
    sub2.set_yticks([0., 0.5, 1., 1.5])
    sub2.set_ylabel('$p\,(\;\mathrm{log}\; \mathrm{SSFR}\;)$', fontsize=20)
    # mass bin 
    sub2.text(0.5, 0.9, '$'+str(mrange[0])+'< \mathrm{log}\, M_* <'+str(mrange[1])+'$',
            ha='center', va='center', transform=sub2.transAxes, fontsize=20)
    sub2.text(0.1, 0.33, '$f_\mathrm{SFMS}='+str(round(fit_fsfms[i_fit],2))+'$',
            ha='left', va='center', transform=sub2.transAxes, fontsize=20)
    fig.subplots_adjust(wspace=.3)
    fig.savefig(''.join([UT.tex_dir(), 'figs/groupcat.pdf']), bbox_inches='tight', dpi=150) 
    plt.close()
    return None


def fQ_fSFMS(): 
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
    coeff = np.polyfit(fit_logm, fit_fsfms, 1)

    # output f_SFMS to data (for posterity)
    f = open(''.join([UT.tex_dir(), 'dat/fsfms.dat']), 'w') 
    f.write('### header ### \n') 
    f.write('star-formation main sequence (SFMS) fraction: fraction of galaxies \n') 
    f.write('within a log-normal fit of the SFMS. See paper for details.\n') 
    f.write('best-fit f_SFMS = '+str(round(coeff[0], 3))+' log M* + '+str(round(coeff[1],3))+'\n')
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
    fsfms_bf = sub.plot(marr, 1-(coeff[0]*marr + coeff[1]), c=pretty_colors[1], lw=2, ls='-')
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


def SFHmodel(nsnap0=15):
    ''' Figure that illustrates the SFH of galaxies. 
    Two panel plot. Panel a) SFH of a galaxy plotted alongside SFMS 
    '''
    pretty_colors = prettycolors()  
    fig = plt.figure(figsize=(10,5))
    # log SFR - log M* galaxy evolution 
    sub1 = fig.add_subplot(121)
    # Delta log SFR(t) evolution 
    sub2 = fig.add_subplot(122)
    sub2.fill_between([5., 14.], [0.3, 0.3], [-0.3, -0.3], color='k', alpha=0.15, linewidth=0)
    sub2.fill_between([5., 14.], [0.6, 0.6], [-0.6, -0.6], color='k', alpha=0.05, linewidth=0)
    
    for i_m, method in enumerate(['randomSFH', 'randomSFH_long']): 
        subcat, eev = ABC.model(method, np.array([1.35, 0.6]), nsnap0=nsnap0, 
                downsampled='14', sigma_smhm=0.2, forTests=True)

        # randomly pick a galaxy that match the below criteria
        isSF = np.where(subcat['gclass'] == 'sf') 
        eligible = np.where((subcat['nsnap_start'][isSF] == nsnap0) & 
                (subcat['weights'][isSF] > 0.) & 
                (subcat['m.star0'][isSF] > 10.15+0.3*float(i_m)) & 
                (subcat['m.star0'][isSF] < 10.25+0.3*float(i_m)))
        i_random = np.random.choice(eligible[0], size=1)
        i_gal = isSF[0][i_random]

        # track back it's M* and SFR
        mstar_hist, sfr_hist = [subcat['m.star0'][i_gal][0]], [subcat['sfr0'][i_gal][0]] 
        
        for isnap in range(2,nsnap0)[::-1]: 
            mstar_hist.append(subcat['m.star.snap'+str(isnap)][i_gal][0])
            sfr_hist.append(subcat['sfr.snap'+str(isnap)][i_gal][0])
        mstar_hist.append(subcat['m.star'][i_gal][0]) 
        sfr_hist.append(subcat['sfr'][i_gal][0]) 
        sfr_hist = np.array(sfr_hist)
        
        # SFMS 
        sfr_sfms = [Obvs.SSFR_SFMS(mstar_hist[0], UT.z_nsnap(nsnap0), 
            theta_SFMS=subcat['theta_sfms']) + mstar_hist[0]]
        for ii, isnap in enumerate(range(2,nsnap0)[::-1]): 
            sfr_sfms.append(Obvs.SSFR_SFMS(mstar_hist[ii+1], UT.z_nsnap(isnap), 
                theta_SFMS=subcat['theta_sfms']) + mstar_hist[ii+1])
        sfr_sfms.append(Obvs.SSFR_SFMS(mstar_hist[-1], UT.z_nsnap(1), 
                theta_SFMS=subcat['theta_sfms']) + mstar_hist[-1])
        sfr_sfms = np.array(sfr_sfms)[:,0]
        f_sfms = interp1d(mstar_hist, sfr_sfms, kind='slinear') #smooth
        t_mstar = interp1d(mstar_hist, UT.t_nsnap(range(1,nsnap0+1)[::-1]))

        marr = np.linspace(mstar_hist[0], mstar_hist[-1], 200)
        sub1.plot(marr, f_sfms(marr), c='k', lw=1)
        #sub1.plot(mstar_hist, sfr_hist, c=pretty_colors[2])

        def dlogSFR_t(tt):
            tsteps = eev.tsteps[i_random][0]
            dlogSFR_amp = eev.dlogSFR_amp[i_random][0]
            ishift = np.abs(tsteps - tt).argmin()
            closest = tsteps[ishift]
            if closest > tt: 
                ishift -= 1 
            dlogsfr = dlogSFR_amp[ishift]
            return dlogsfr

        sub1.plot(marr, f_sfms(marr)+np.array([dlogSFR_t(tt) for tt in t_mstar(marr)]), 
                c=pretty_colors[2*i_m+1])
        # plot SFMS(M*, z)
        m_arr = np.linspace(9., 12., 100)
        zs = np.arange(0., 1.2, 0.25)
        zs[0] = 0.05
        for z in zs: 
            sub1.plot(m_arr, Obvs.SSFR_SFMS(m_arr, z, theta_SFMS=subcat['theta_sfms']) + m_arr,
                    c='k', ls=':', lw=0.75)
            sub1.text(10.05, Obvs.SSFR_SFMS(10.05, z, theta_SFMS=subcat['theta_sfms'])+10.05, 
                    '$z = '+str(z)+'$', 
                    rotation=0.5*np.arctan(subcat['theta_sfms']['mslope'])*180./np.pi, 
                    fontsize=12, va='bottom')
        
        xx, yy = [], []
        for i in range(len(eev.dlogSFR_amp[i_random][0])-1):
            xx.append(eev.tsteps[i_random][0][i]) 
            yy.append(eev.dlogSFR_amp[i_random][0][i])
            xx.append(eev.tsteps[i_random][0][i+1]) 
            yy.append(eev.dlogSFR_amp[i_random][0][i])
        if i_m == 0: 
            sub2.plot([UT.t_nsnap(nsnap0), UT.t_nsnap(1)], [0.,0.], ls='--', c='k')
        if method == 'randomSFH': 
            lbl = '$t_\mathrm{duty} = 0.5\,\mathrm{Gyr}$'
        elif method == 'randomSFH_long':
            lbl = '$t_\mathrm{duty} = 5\,\mathrm{Gyr}$'
            
        sub2.plot(xx, yy, c=pretty_colors[2*i_m+1], label=lbl)

    sub1.set_xlim([10., 11.])
    sub1.set_xticks([10., 10.5, 11.])
    sub1.set_xlabel('log $(\; M_*\; [M_\odot]\;)$', fontsize=25)
    sub1.set_ylim([-1., 1.5])
    sub1.set_ylabel('log $(\; \mathrm{SFR}\; [M_\odot/\mathrm{yr}]\;)$', fontsize=25)

    sub2.set_xlim([UT.t_nsnap(nsnap0), UT.t_nsnap(1)]) 
    sub2.set_xlabel('$t_\mathrm{cosmic}\;[\mathrm{Gyr}]$', fontsize=25)
    sub2.set_ylim([-1., 1.]) 
    sub2.set_yticks([-1., -0.5, 0., 0.5, 1.])
    sub2.set_ylabel('$\Delta$ log $(\;\mathrm{SFR}\;[M_\odot/\mathrm{yr}])$', fontsize=25)
    sub2.legend(loc='lower right', prop={'size':15})
    #sub2.yaxis.tick_right()
    #sub2.yaxis.set_label_position("right")
    fig.subplots_adjust(wspace=0.4)
    fig.savefig(''.join([UT.tex_dir(), 'figs/sfh_pedagogical.pdf']), bbox_inches='tight', dpi=150) 
    return None 


if __name__=="__main__": 
    groupcatSFMS(mrange=[10.6,10.8])
    #fQ_fSFMS()
    #SFHmodel(nsnap0=15)
