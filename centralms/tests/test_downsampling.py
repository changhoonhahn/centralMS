'''

Test to make sure that the downsampling of the catalog
produces (more or less) the same results as the full catalog

'''


import env

import numpy as np 

import catalog as Cat
import evolver as Evol
import observables as Obvs
import util as UT

import matplotlib.pyplot as plt 
import bovy_plot as bovy
import corner as DFM
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors

def test_EvolverInitiate_downsample(test, nsnap, nsnap0=20, downsampled=None): 
    ''' Tests for Initiate method in Evolver for specified nsnap snapshot.
    '''
    if nsnap > nsnap0: 
        raise ValueError('nsnap has to be less than or equal to nsnap0')
    if downsampled is None: 
        raise ValueError('the whole point of this function is to test downsampling...')

    # load in Subhalo Catalog (pure centrals)
    subhist = Cat.PureCentralHistory(nsnap_ancestor=nsnap0)
    subcat = subhist.Read(downsampled=None) # full sample
    subcat_down = subhist.Read(downsampled=downsampled) # downsampled

    theta = Evol.defaultTheta('constant_offset') # load in generic theta (parameters)

    eev = Evol.Evolver(subcat, theta, nsnap0=nsnap0)
    eev.Initiate()

    eev_down = Evol.Evolver(subcat_down, theta, nsnap0=nsnap0)
    eev_down.Initiate()

    if test ==  'pssfr': # calculate P(SSFR) 
        obv_ssfr = Obvs.Ssfr()
        
        # full sample P(ssfr)
        started = np.where(subcat['nsnap_start'] == nsnap)

        ssfr_mids, pssfrs = obv_ssfr.Calculate(
                subcat['m.star0'][started], 
                subcat['sfr0'][started]-subcat['m.star0'][started], 
                weights=subcat['weights'][started])
        x_ssfrs = obv_ssfr.ssfr_bin_edges

        # down-sample P(ssfr)
        started = np.where(subcat_down['nsnap_start'] == nsnap)

        ssfr_mids, pssfrs_down = obv_ssfr.Calculate(
                subcat_down['m.star0'][started], 
                subcat_down['sfr0'][started] - subcat_down['m.star0'][started], 
                weights=subcat_down['weights'][started])
        x_ssfrs_down = obv_ssfr.ssfr_bin_edges

        fig = plt.figure(figsize=(20, 5))
        bkgd = fig.add_subplot(111, frameon=False)

        panel_mass_bins = [[9.7, 10.1], [10.1, 10.5], [10.5, 10.9], [10.9, 11.3]]
        for i_m, mass_bin in enumerate(panel_mass_bins): 
            sub = fig.add_subplot(1, 4, i_m+1)

            # plot P(SSFR) full-sample
            x_bar, y_bar = UT.bar_plot(x_ssfrs[i_m], pssfrs[i_m])
            sub.plot(x_bar, y_bar, lw=2, ls='-', c='k') 
            
            # plot P(SSFR) full-sample
            x_bar, y_bar = UT.bar_plot(x_ssfrs_down[i_m], pssfrs_down[i_m])
            sub.plot(x_bar, y_bar, lw=3, ls='--', c='k') 
            
            # mark the SSFR of SFMS and Quiescent peak 
            sub.vlines(Obvs.SSFR_SFMS(0.5 * np.sum(mass_bin), UT.z_nsnap(nsnap), theta_SFMS=theta['sfms']), 0., 1.7, 
                    color='b', linewidth=2, linestyle='-')
            sub.vlines(Obvs.SSFR_Qpeak(0.5 * np.sum(mass_bin)), 0., 1.7, 
                    color='r', linewidth=2, linestyle='-')

            massbin_str = ''.join([ 
                r'$\mathtt{log \; M_{*} = [', 
                str(mass_bin[0]), ',\;', 
                str(mass_bin[1]), ']}$'
                ])
            sub.text(-12., 1.4, massbin_str, fontsize=20)
        
            # x-axis
            sub.set_xlim([-13., -8.])
            # y-axis 
            sub.set_ylim([0.0, 1.7])
            sub.set_yticks([0.0, 0.5, 1.0, 1.5])
            if i_m == 0: 
                sub.set_ylabel(r'$\mathtt{P(log \; SSFR)}$', fontsize=25) 
            else: 
                sub.set_yticklabels([])
            #ax = plt.gca()
            #leg = sub.legend(bbox_to_anchor=(-8.5, 1.55), loc='upper right', prop={'size': 20}, borderpad=2, 
            #        bbox_transform=ax.transData, handletextpad=0.5)
        
        bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        bkgd.set_xlabel(r'$\mathtt{log \; SSFR \;[yr^{-1}]}$', fontsize=25) 
        plt.show()

    elif test == 'fq': # calculate quiescent fraction 
        obv_fq = Obvs.Fq()

        pretty_colors = prettycolors()

        fig = plt.figure(figsize=(6,6))
        sub = fig.add_subplot(111)
        
        print 'Full Sample'
        started = np.where(subcat['nsnap_start'] == nsnap)
        print len(started[0]), ' galaxies'
        print np.sum(subcat['weights'][started])

        m_mid, fq, counts = obv_fq.Calculate(
                mass=subcat['m.star0'][started], 
                sfr=subcat['sfr0'][started], 
                z=UT.z_nsnap(nsnap), weights= subcat['weights'][started], theta_SFMS=theta['sfms'], counts=True)

        print 'Down Sample'
        started = np.where(subcat_down['nsnap_start'] == nsnap)
        print len(started[0]), ' galaxies'
        print np.sum(subcat['weights'][started])

        m_mid_down, fq_down, counts_down = obv_fq.Calculate(
                mass=subcat_down['m.star0'][started], 
                sfr=subcat_down['sfr0'][started], 
                z=UT.z_nsnap(nsnap), weights= subcat_down['weights'][started], theta_SFMS=theta['sfms'], counts=True)

        cc = pretty_colors[nsnap]
        #sub.scatter(m_mid, fq, c=cc, s=10)
        sub.plot(m_mid, fq, c=cc, lw=2)
        sub.plot(m_mid_down, fq_down, c=cc, lw=3, ls='--')
        sub.plot(m_mid, obv_fq.model(m_mid, UT.z_nsnap(nsnap), lit='cosmos_tinker'), c=cc, ls=':')
        
        #for i in range(len(m_mid)): 
        #    sub.text(m_mid[i], 0.05+fq[i], str(counts[i]))
        plt.show()
    
    elif test == 'smf_evol': # check the SMF evolution of the SF population
        fig = plt.figure(figsize=(7,7))
        sub = fig.add_subplot(111)
        
        snaps = [] # pick a handful of snapshots 
        for ii in range(2, nsnap0+1): 
            if (ii-1)%5 == 0: 
                snaps.append(ii)
        snaps.append(nsnap0) 

        for n in snaps[::-1]: 
            # SF population at snapshot (full sample)
            pop_sf = np.where(
                    (subcat['gclass0'] == 'star-forming') & 
                    (subcat['nsnap_quench'] <= n) & 
                    (subcat['weights'] > 0.)
                    )

            smf_sf = Obvs.getMF(
                    subcat['snapshot'+str(n)+'_m.sham'][pop_sf], 
                    weights=subcat['weights'][pop_sf])
    
            sub.plot(smf_sf[0], smf_sf[1], lw=2, c='k', alpha=0.05 * (21. - n))#, label='Snapshot '+str(n))
    
            pop_sf = np.where(
                    (subcat_down['gclass0'] == 'star-forming') & 
                    (subcat_down['nsnap_quench'] <= n) & 
                    (subcat_down['weights'] > 0.)
                    )

            smf_sf = Obvs.getMF(
                    subcat_down['snapshot'+str(n)+'_m.sham'][pop_sf], 
                    weights=subcat_down['weights'][pop_sf])
    
            sub.plot(smf_sf[0], smf_sf[1], lw=3, ls='--', c='k', alpha=0.05 * (21. - n))#, label='Snapshot '+str(n))

        # nsnap = 1 full sample
        pop_sf = np.where(
                    (subcat['gclass'] == 'star-forming') & 
                    (subcat['weights'] > 0.)
                    )
        smf_sf = Obvs.getMF(
                subcat['m.sham'][pop_sf], 
                weights=subcat['weights'][pop_sf])
        sub.plot(smf_sf[0], smf_sf[1], lw=3, c='k', ls='-', label='Snapshot 1')
        
        # nsnap = 1 down sample
        pop_sf = np.where(
                    (subcat_down['gclass'] == 'star-forming') & 
                    (subcat_down['weights'] > 0.)
                    )
        smf_sf = Obvs.getMF( subcat_down['m.sham'][pop_sf], weights=subcat_down['weights'][pop_sf])
        sub.plot(smf_sf[0], smf_sf[1], lw=3, c='k', ls='--')

        sub.set_xlim([6., 12.])
        sub.set_xlabel('Stellar Masses $(\mathcal{M}_*)$', fontsize=25)
        sub.set_ylim([1e-5, 10**-1.5])
        sub.set_yscale('log')
        sub.set_ylabel('$\Phi$', fontsize=25)
        sub.legend(loc='upper right') 
        plt.show()
    
    elif test == 'sfms': # check the SFMS of the initial SFRs of the full vs down-samples
    
        fig = plt.figure(figsize=(7,7))
        sub = fig.add_subplot(111)

        # SFMS of the full sample 
        started = np.where(subcat['nsnap_start'] == nsnap)

        DFM.hist2d(subcat['m.star0'][started], subcat['sfr0'][started], weights=subcat['weights'][started], 
                levels=[0.68, 0.95], range=[[6., 12.], [-4., 2.]], color='#1F77B4', 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub) 

        # SFMS of the down sample 
        started = np.where(subcat_down['nsnap_start'] == nsnap)
        DFM.hist2d(subcat_down['m.star0'][started], subcat_down['sfr0'][started], weights=subcat_down['weights'][started], 
                levels=[0.68, 0.95], range=[[6., 12.], [-4., 2.]], color='#FF7F0E', 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub) 
        plt.show()



    return None


def test_EvolverEvolve_downsample(test, nsnap0=20, downsampled=None): 
    ''' Tests for Evolve method in Evolver
    '''
    if downsampled is None: 
        raise ValueError('the whole point of this function is to test downsampling...')

    # load in generic theta (parameter values)
    theta = Evol.defaultTheta() 

    # load in Subhalo Catalog (pure centrals)
    subhist = Cat.PureCentralHistory(nsnap_ancestor=nsnap0)
    
    subcat = subhist.Read(downsampled=None) # full sample
    eev = Evol.Evolver(subcat, theta, nsnap0=nsnap0)
    eev.Initiate()
    eev.Evolve() 
    subcat = eev.SH_catalog

    subcat_down = subhist.Read(downsampled=downsampled) # downsampled
    eev_down = Evol.Evolver(subcat_down, theta, nsnap0=nsnap0)
    eev_down.Initiate()
    eev_down.Evolve() 
    subcat_down = eev_own.SH_catalog

    raise ValueError()
    pretty_colors = prettycolors() 
    if test == 'smf': 
        isSF = np.where(subcat['gclass'] == 'star-forming')
        
        fig = plt.figure(figsize=(7,7))
        sub = fig.add_subplot(111)

        for n in range(2, 21)[::-1]: 
            # identify SF population at snapshot
            smf_sf = Obvs.getMF(
                    subcat['snapshot'+str(n)+'_m.star'][isSF], 
                    weights=subcat['weights'][isSF])

            sub.plot(smf_sf[0], smf_sf[1], lw=2, c='b', alpha=0.05 * (21. - n))        

        smf_sf_msham0 = Obvs.getMF(subcat['m.star0'][isSF], weights=subcat['weights'][isSF])
        sub.plot(smf_sf_msham0[0], smf_sf_msham0[1], lw=3, c='k', ls='--')

        smf_sf_msham = Obvs.getMF(subcat['m.sham'][isSF], weights=subcat['weights'][isSF])
        sub.plot(smf_sf_msham[0], smf_sf_msham[1], lw=3, c='k', ls='--', label='SHAM')
        
        print np.sum(subcat['m.star'][isSF] < 0.)
        print subcat['m.star'][isSF].min(), subcat['m.star'][isSF].max()

        #raise ValueError
        smf_sf = Obvs.getMF(subcat['m.star'][isSF], weights=subcat['weights'][isSF])
        sub.plot(smf_sf[0], smf_sf[1], lw=3, c='b', ls='-', label='Integrated')

        sub.set_xlim([6., 12.])
        sub.set_xlabel('Stellar Masses $(\mathcal{M}_*)$', fontsize=25)
        sub.set_ylim([1e-6, 10**-1.75])
        sub.set_yscale('log')
        sub.set_ylabel('$\Phi$', fontsize=25)
        sub.legend(loc='upper right') 
        plt.show()

    elif test == 'smf_comp':  # SMF of composition
        isSF = np.where(subcat['gclass'] == 'star-forming')[0]
        
        fig = plt.figure(figsize=(15,7))
        sub = fig.add_subplot(121)
        
        #smf_sf_msham0 = Obvs.getMF(subcat['m.star0'][isSF], weights=subcat['weights'][isSF])
        #sub.plot(smf_sf_msham0[0], smf_sf_msham0[1], lw=3, c='k', ls='--')
    
        m0s = subcat['m.star0'][isSF]
        mlow = np.arange(m0s.min(), m0s.max(), 0.5)

        for i_m in range(len(mlow)): 
            inMbin = np.where((subcat['m.star0'][isSF] > mlow[i_m]) & (subcat['m.star0'][isSF] < mlow[i_m] + 0.5))

            smf_sf = Obvs.getMF(subcat['m.star'][isSF[inMbin]], weights=subcat['weights'][isSF[inMbin]])

            if i_m == 0: 
                smf_sf0 = np.zeros(len(smf_sf[0]))
                smf_sf1 = smf_sf[1]
            else: 
                smf_sf1 = smf_sf0 + smf_sf[1]

            mbin_label = ''.join([str(mlow[i_m])+', '+str(mlow[i_m] + 0.5)]) 

            sub.fill_between(smf_sf[0], smf_sf0, smf_sf1, 
                    facecolor=pretty_colors[i_m % 20], edgecolor=None, lw=0)#, label=mbin_label)

            smf_sf0 = smf_sf1

        smf_sf = Obvs.getMF(subcat['m.star'][isSF], weights=subcat['weights'][isSF])
        sub.plot(smf_sf[0], smf_sf[1], lw=3, c='k', ls='-')

        sub.set_xlim([6., 12.])
        sub.set_xlabel('Stellar Masses $(\mathcal{M}_*)$', fontsize=25)
        sub.set_ylim([1e-6, 10**-1.75])
        sub.set_yscale('log')
        sub.set_ylabel('$\Phi$', fontsize=25)
        #sub.legend(loc='upper right') 

        sub = fig.add_subplot(122)
        
        m0s = subcat['m.star0'][isSF]
        mlow = np.arange(m0s.min(), m0s.max(), 0.5)

        for i_m in range(len(mlow)): 
            inMbin = np.where((subcat['m.star0'][isSF] > mlow[i_m]) & (subcat['m.star0'][isSF] < mlow[i_m] + 0.5))

            smf_sf = Obvs.getMF(subcat['m.sham'][isSF[inMbin]], weights=subcat['weights'][isSF[inMbin]])

            if i_m == 0: 
                smf_sf0 = np.zeros(len(smf_sf[0]))
                smf_sf1 = smf_sf[1]
            else: 
                smf_sf1 = smf_sf0 + smf_sf[1]

            mbin_label = ''.join([str(mlow[i_m])+', '+str(mlow[i_m] + 0.5)]) 

            sub.fill_between(smf_sf[0], smf_sf0, smf_sf1, 
                    facecolor=pretty_colors[i_m % 20], edgecolor=None, lw=0)#, label=mbin_label)

            smf_sf0 = smf_sf1

        smf_sf = Obvs.getMF(subcat['m.sham'][isSF], weights=subcat['weights'][isSF])
        sub.plot(smf_sf[0], smf_sf[1], lw=3, c='k', ls='-')

        sub.set_xlim([6., 12.])
        sub.set_xlabel('Stellar Masses $(\mathcal{M}_{SHAM})$', fontsize=25)
        sub.set_ylim([1e-6, 10**-1.75])
        sub.set_yscale('log')
        plt.show()

    elif test == 'pssfr':
        obv_ssfr = Obvs.Ssfr()
        
        isSF = np.where(subcat['gclass'] == 'star-forming')
        ssfr_bin_mids, ssfr_dists0 = obv_ssfr.Calculate(subcat['m.star0'][isSF], 
                subcat['sfr0'][isSF]-subcat['m.star0'][isSF], 
                subcat['weights'][isSF])
    
        ssfr_bin_mids, ssfr_dists = obv_ssfr.Calculate(subcat['m.star'][isSF], 
                subcat['sfr'][isSF]-subcat['m.star'][isSF], 
                subcat['weights'][isSF])

        fig = plt.figure(figsize=(20, 5))
        bkgd = fig.add_subplot(111, frameon=False)

        panel_mass_bins = [[9.7, 10.1], [10.1, 10.5], [10.5, 10.9], [10.9, 11.3]]
        for i_m, mass_bin in enumerate(panel_mass_bins): 
            sub = fig.add_subplot(1, 4, i_m+1)

            sub.plot(ssfr_bin_mids[i_m], ssfr_dists0[i_m], 
                    lw=3, ls='--', c='b')
            sub.plot(ssfr_bin_mids[i_m], ssfr_dists[i_m], 
                    lw=3, ls='-', c='k')
            
            # mark the SSFR of SFMS and Quiescent peak 
            sub.vlines(Obvs.SSFR_SFMS(0.5 * np.sum(mass_bin), UT.z_nsnap(1), theta_SFMS=theta['sfms']), 0., 1.7, 
                    color='b', linewidth=3)
            sub.vlines(Obvs.SSFR_Qpeak(0.5 * np.sum(mass_bin)), 0., 1.7, 
                    color='r', linewidth=3)

            massbin_str = ''.join([ 
                r'$\mathtt{log \; M_{*} = [', 
                str(mass_bin[0]), ',\;', 
                str(mass_bin[1]), ']}$'
                ])
            sub.text(-12., 1.4, massbin_str, fontsize=20)
        
            # x-axis
            sub.set_xlim([-13., -8.])
            # y-axis 
            sub.set_ylim([0.0, 1.7])
            sub.set_yticks([0.0, 0.5, 1.0, 1.5])
            if i_m == 0: 
                sub.set_ylabel(r'$\mathtt{P(log \; SSFR)}$', fontsize=25) 
            else: 
                sub.set_yticklabels([])
            
            ax = plt.gca()
            leg = sub.legend(bbox_to_anchor=(-8.5, 1.55), loc='upper right', prop={'size': 20}, borderpad=2, 
                    bbox_transform=ax.transData, handletextpad=0.5)
        
        bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        bkgd.set_xlabel(r'$\mathtt{log \; SSFR \;[yr^{-1}]}$', fontsize=25) 
        plt.show()

    elif test == 'smhmr': # stellar mass to halo mass relation 
        isSF = np.where(subcat['gclass'] == 'star-forming')

        smhmr = Obvs.Smhmr()
        m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat['m.star'][isSF], subcat['halo.m'][isSF])

        fig = plt.figure()
        sub = fig.add_subplot(111)
        
        sub.errorbar(m_mid, mu_mhalo, yerr=sig_mhalo)
        sub.fill_between(m_mid, mu_mhalo - 0.2, mu_mhalo + 0.2, color='k', alpha=0.25, linewidth=0, edgecolor=None)
    
        sub.set_xlim([8., 12.])
        sub.set_xlabel('Stellar Mass $(\mathcal{M}_*)$', fontsize=25)
        sub.set_ylabel('Halo Mass $(\mathcal{M}_{halo})$', fontsize=25)
        
        plt.show()

    elif test == 'sfms': 
        isSF = np.where(subcat['gclass'] == 'star-forming')

        bovy.scatterplot(subcat['m.star'][isSF], subcat['sfr'][isSF], scatter=True, s=2, 
                xrange=[8., 12.], yrange=[-4., 3.],
                xlabel='\mathtt{log\;M_*}', ylabel='\mathtt{log\;SFR}')

        m_arr = np.arange(8., 12.1, 0.1) 
        plt.plot(m_arr, Obvs.SSFR_SFMS(m_arr, UT.z_nsnap(1), theta_SFMS=theta['sfms']) + m_arr, c='r', lw=2, ls='-')
        plt.plot(m_arr, Obvs.SSFR_SFMS(m_arr, UT.z_nsnap(1), theta_SFMS=theta['sfms']) + m_arr - 0.3, c='r', lw=2, ls='-.')
        plt.plot(m_arr, Obvs.SSFR_SFMS(m_arr, UT.z_nsnap(1), theta_SFMS=theta['sfms']) + m_arr + 0.3, c='r', lw=2, ls='-.')

        plt.show()

    elif test == 'delMstar':  # difference between sham M* and integrated M* 
        isSF = np.where(subcat['gclass'] == 'star-forming')

        delMstar = subcat['m.star'][isSF] - subcat['m.sham'][isSF]  # Delta M*

        bovy.scatterplot(subcat['m.star'][isSF], delMstar, scatter=True, s=2, 
                xrange=[8., 12.], yrange=[-4., 4.], xlabel='\mathtt{log\;M_*}', ylabel='\mathtt{log\;M_* - log\;M_{sham}}')

        plt.show()

    elif test == 'delMgrowth': 
        isSF = np.where(subcat['gclass'] == 'star-forming')

        bovy.scatterplot(subcat['m.star'][isSF], subcat['m.star'][isSF] - subcat['m.star0'][isSF], scatter=True, s=2, 
                xrange=[8., 12.], yrange=[-4., 4.], 
                xlabel=r'{\rm Integrated}\;\mathtt{log\;M_*}', ylabel='\mathtt{log\;M_* - log\;M_0}')

        bovy.scatterplot(subcat['m.sham'][isSF], subcat['m.sham'][isSF] - subcat['m.star0'][isSF], scatter=True, s=2, 
                xrange=[8., 12.], yrange=[-4., 4.], 
                xlabel=r'{\rm SHAM}\;\mathtt{log\;M_*}', ylabel='\mathtt{log\;M_* - log\;M_0}')

        plt.show()

    elif test == 'sfh_sfms':  # plot the SFH as a function of time 
        # first pick a random SF galaxy
        isSF = np.where((subcat['gclass'] == 'star-forming') & (subcat['nsnap_start'] == 20))[0]

        m_bin = np.arange(9.0, 12., 0.5)  
        i_bin = np.digitize(subcat['m.star0'][isSF], m_bin)
        
        fig = plt.figure()
        sub = fig.add_subplot(111)

        for i in np.unique(i_bin): 
            i_rand = np.random.choice(np.where(i_bin == i)[0], size=1)[0]
            
            sfrs = [subcat['sfr0'][isSF[i_rand]]]
            mstars = [subcat['m.star0'][isSF[i_rand]]]
            for nn in range(2, 20)[::-1]: 
                sfrs.append(subcat['snapshot'+str(nn)+'_sfr'][isSF[i_rand]])
                mstars.append(subcat['snapshot'+str(nn)+'_m.star'][isSF[i_rand]])
            sfrs.append(subcat['sfr'][isSF[i_rand]])
            mstars.append(subcat['m.star'][isSF[i_rand]])
            sub.scatter(mstars, sfrs, c=pretty_colors[i], lw=0)

        sub.set_xlim([9.0, 13.])
        sub.set_xlabel('Stellar Mass $(\mathcal{M}_*)$', fontsize=25)
        sub.set_ylabel('log SFR', fontsize=25)
        plt.show()

    elif test == 'sfh':  # plot the SFH as a function of time 
        # first pick a random SF galaxy
        isSF = np.where((subcat['gclass'] == 'star-forming') & (subcat['nsnap_start'] == 20))[0]

        m_bin = np.arange(9.0, 12.5, 0.5)  
        i_bin = np.digitize(subcat['m.star0'][isSF], m_bin)
        
        fig = plt.figure()
        sub = fig.add_subplot(111)

        for i in np.unique(i_bin): 
            i_rand = np.random.choice(np.where(i_bin == i)[0], size=1)[0]
            
            dsfrs = [subcat['sfr0'][isSF[i_rand]] - (Obvs.SSFR_SFMS(
                subcat['m.star0'][isSF[i_rand]], UT.z_nsnap(20), 
                theta_SFMS=eev.theta_sfms) + subcat['m.star0'][isSF[i_rand]])[0]]

            for nn in range(2, 20)[::-1]: 
                M_nn = subcat['snapshot'+str(nn)+'_m.star'][isSF[i_rand]]
                mu_sfr = Obvs.SSFR_SFMS(M_nn, UT.z_nsnap(nn), theta_SFMS=eev.theta_sfms) + M_nn
                dsfrs.append(subcat['snapshot'+str(nn)+'_sfr'][isSF[i_rand]] - mu_sfr[0])

            mu_sfr = Obvs.SSFR_SFMS(subcat['m.star'][isSF[i_rand]], 
                    UT.z_nsnap(1), theta_SFMS=eev.theta_sfms) + subcat['m.star'][isSF[i_rand]]
            dsfrs.append(subcat['sfr'][isSF[i_rand]] - mu_sfr[0]) 
            sub.plot(UT.t_nsnap(range(1,21)[::-1]), dsfrs, c=pretty_colors[i], lw=2)
        #sub.set_xlim([9.0, 13.])
        sub.set_xlabel('$t_{cosmic}$', fontsize=25)
        sub.set_ylabel('$\Delta$log SFR', fontsize=25)
        plt.show()

    return None


if __name__=="__main__": 
    #test_RandomStep_timescale(sig_smhm=0.2, nsnap_ancestor=15)
    #EvolverPlots('constant_offset')
    #EvolverPlots('corr_constant_offset')
    #EvolverPlots('random_step', nsnap0=15)
    #test_AssemblyBias(0.3, nsnap0=15)
    #test_Evolver_AssemblyBias(0.3, nsnap0=15)
    #EvolverPlots('constant_offset', nsnap0=15)
    #EvolverPlots('random_step', nsnap0=15)
    #EvolverPlots('random_step_fluct', nsnap0=15)
    #EvolverPlots('random_step_abias2', nsnap0=15)
    #test_Evolver_logSFRinitiate('random_step_abias', nsnap0=15)
    #test_Evolver_logSFRinitiate('random_step_fluct', nsnap0=15)
    #test_Evolver_ODEsolver('random_step_fluct', nsnap0=15)
    #test_EvolverEvolve('smhmr')
    for i in [15, 10, 5, 1]: 
        test_EvolverInitiate_downsample('sfms', i, nsnap0=15, downsampled='14')
    #test_EvolverInitiate('pssfr', 15)
    #test_assignSFRs() 
