#!/usr/bin/env python
''' 

Make figures for paper 


'''
import numpy as np 
from scipy.interpolate import interp1d
import corner as DFM

import util as UT
import abcee as ABC
import catalog as Cat
import observables as Obvs

import matplotlib as mpl 
import matplotlib.pyplot as plt 
from cycler import cycler
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors
from matplotlib import rcParams

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5

def groupcatSFMS(mrange=[10.2,10.4]): 
    '''Figure of the z~0 group catalog. 
    Panel a) SFR-M* relation 
    Panel b) P(SSFR) with SFMS fitting 
    '''
    # Read in Jeremy's group catalog  with Mr_cut = -18
    gc = Cat.Observations('group_catalog', Mrcut=18, position='central')
    gc_cat = gc.Read() 
    fig = plt.figure(figsize=(10,5)) 

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
    sub1.set_ylabel('log$(\; \mathrm{SFR}\; [M_\odot/yr]\;)$', fontsize=20)
    sub1.text(9.2, 1., 'SDSS central galaxies', fontsize=20) 

    # P(log SSFR) 
    sub2 = fig.add_subplot(1,2,2)
    inmbin = np.where((gc_cat['mass'] > mrange[0]) & (gc_cat['mass'] < mrange[1]))
    bedge, pp = np.histogram(gc_cat['ssfr'][inmbin], range=[-14., -9.], bins=32, normed=True)
    print bedge.sum()
    pssfr = UT.bar_plot(pp, bedge)
    sub2.plot(pssfr[0], pssfr[1], c='k', lw=2) 
    sub2.set_xlim([-9.5, -13.25]) 
    sub2.set_xticks([-10., -11., -12., -13.])
    sub2.set_xlabel('log$(\; \mathrm{SSFR}\; [yr^{-1}]\;)$', fontsize=20)
    sub2.set_ylim([0., 1.1]) 
    sub2.set_ylabel('$p(\;log\; \mathrm{SSFR}\;)$', fontsize=20)
    sub2.text(-10., 0.8, '$f_\mathrm{SFMS}=$', fontsize=20) 
    fig.subplots_adjust(wspace=.3)
    fig.savefig(''.join([UT.tex_dir(), 'figs/groupcat.png']), bbox_inches='tight') 
    plt.close()
    return None


def SFHmodel(nsnap0=15):
    ''' Figure that illustrates the SFH of galaxies. 
    Two panel plot. Panel a) SFH of a galaxy plotted alongside SFMS 
    '''
    prettyplot() 
    pretty_colors = prettycolors()  
    fig = plt.figure(figsize=(15,7))
    # log SFR - log M* galaxy evolution 
    sub1 = fig.add_subplot(121)
    # Delta log SFR(t) evolution 
    sub2 = fig.add_subplot(122)
    
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
        for z in np.arange(0.1, 1.25, 0.25): 
            sub1.plot(m_arr, Obvs.SSFR_SFMS(m_arr, z, theta_SFMS=subcat['theta_sfms']) + m_arr,
                    c='k', ls=':', lw=0.75)
            sub1.text(10.05, Obvs.SSFR_SFMS(10.05, z, theta_SFMS=subcat['theta_sfms'])+10.12, 
                    '$\mathtt{z = '+str(z)+'}$', 
                    rotation=0.4*np.arctan(subcat['theta_sfms']['mslope'])*180./np.pi, 
                    fontsize=12)
        
        xx, yy = [], []
        for i in range(len(eev.dlogSFR_amp[i_random][0])-1):
            xx.append(eev.tsteps[i_random][0][i]) 
            yy.append(eev.dlogSFR_amp[i_random][0][i])
            xx.append(eev.tsteps[i_random][0][i+1]) 
            yy.append(eev.dlogSFR_amp[i_random][0][i])
        if i_m == 0: 
            sub2.plot([UT.t_nsnap(nsnap0), UT.t_nsnap(1)], [0.,0.], ls='--', c='k')
        sub2.plot(xx, yy, c=pretty_colors[2*i_m+1])

    sub1.set_xlim([10., 11.])
    sub1.set_xlabel('$\mathtt{log(\; M_*\; [M_\odot]\;)}$', fontsize=25)
    sub1.set_ylim([-1., 1.5])
    sub1.set_ylabel('$\mathtt{log(\; SFR\; [M_\odot/yr]\;)}$', fontsize=25)

    sub2.set_xlim([UT.t_nsnap(nsnap0), UT.t_nsnap(1)]) 
    sub2.set_xlabel('$\mathtt{t_{cosmic}\;[Gyr]}$', fontsize=25)
    sub2.set_ylim([-1., 1.]) 
    sub2.set_yticks([-1., -0.5, 0., 0.5, 1.])
    sub2.set_ylabel('$\Delta \mathtt{log(\;SFR\;[M_\odot/yr])}$', fontsize=25)
    sub2.yaxis.tick_right()
    sub2.yaxis.set_label_position("right")
    fig.savefig(''.join([UT.tex_dir(), 'figs/sfh_pedagogical.png']), bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    groupcatSFMS()
