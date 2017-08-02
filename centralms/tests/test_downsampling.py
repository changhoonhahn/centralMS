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


def test_Catalog_Downsample(test_type, nsnap0, downsampled='14'): 
    ''' Test the downsampling of the catalog

    ========================================
    Everything looks good
    ========================================
    '''
    subhist = Cat.PureCentralHistory(nsnap_ancestor=nsnap0)

    subcat = subhist.Read() # full sample 

    subcat_down = subhist.Read(downsampled='14')  # downsampled
    
    snaps = [] 
    for ii in range(1, nsnap0+1): 
        if (ii-1)%5 == 0: 
            snaps.append(ii)
    snaps.append(nsnap0) 

    pretty_colors = prettycolors()
    fig = plt.figure(1)
    sub = fig.add_subplot(111)
    
    if test_type == 'hmf': # halo mass function 
        for i in snaps: 
            if i == 1: 
                m_tag = 'halo.m'
            else: 
                m_tag = 'snapshot'+str(i)+'_halo.m'

            shmf = Obvs.getMF(subcat[m_tag], weights=subcat['weights'], m_arr=np.arange(10., 15.5, 0.1))
            sub.plot(shmf[0], shmf[1], c=pretty_colors[i], lw=1, ) 
            
            shmf = Obvs.getMF(subcat_down[m_tag], weights=subcat_down['weights'], m_arr=np.arange(10., 15.5, 0.1))
            sub.plot(shmf[0], shmf[1], c=pretty_colors[i], lw=3, ls='--') 

        # x-axis
        sub.set_xlim([10., 15.])
        # y-axis
        sub.set_yscale("log") 
    
    elif test_type == 'smf': # stellar mass function 
        for i in snaps: 
            if i == 1: 
                m_tag = 'm.star'
            else: 
                m_tag = 'snapshot'+str(i)+'_m.star'

            shmf = Obvs.getMF(subcat[m_tag], weights=subcat['weights'], m_arr=np.arange(8., 12.5, 0.1))
            sub.plot(shmf[0], shmf[1], c=pretty_colors[i], lw=1) 
            
            shmf = Obvs.getMF(subcat_down[m_tag], weights=subcat_down['weights'], m_arr=np.arange(8., 12.5, 0.1))
            sub.plot(shmf[0], shmf[1], c=pretty_colors[i], lw=3, ls='--') 

        # x-axis
        sub.set_xlim([9., 12.])
        # y-axis
        sub.set_yscale("log") 

    elif test_type == 'smhmr': # stellar mass - halo mass relation 
        for i in snaps: 
            if i == 1: 
                hm_tag = 'm.max'
                sm_tag = 'm.star'
            else: 
                hm_tag = 'snapshot'+str(i)+'_m.max'
                sm_tag = 'snapshot'+str(i)+'_m.star'

            smhmr = Obvs.Smhmr()
            m_mid, mu_mstar, sig_mstar, cnts = smhmr.Calculate(subcat[hm_tag], subcat[sm_tag], weights=subcat['weights'])
            sub.plot(m_mid, mu_mstar - sig_mstar, c=pretty_colors[i], lw=1) 
            sub.plot(m_mid, mu_mstar + sig_mstar, c=pretty_colors[i], lw=1) 
            
            m_mid, mu_mstar, sig_mstar, cnts = smhmr.Calculate(subcat_down[hm_tag], subcat_down[sm_tag], 
                    weights=subcat_down['weights'])
            sub.plot(m_mid, mu_mstar - sig_mstar, c=pretty_colors[i], lw=3, ls='--') 
            sub.plot(m_mid, mu_mstar + sig_mstar, c=pretty_colors[i], lw=3, ls='--') 
        
        sub.set_ylim([8., 12.])
    plt.show() 


def test_EvolverInitiate_downsample(test, nsnap, nsnap0=20, downsampled=None): 
    ''' Tests for Initiate method in Evolver for specified nsnap snapshot.

    ========================================
    Everything looks good
    ========================================
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


def test_EvolverEvolve_downsample(test, nsnap0=20, downsampled=None, sfh='constant_offset'): 
    ''' Tests for Evolve method in Evolver

    ========================================
    SMF, P(ssfr), SFMS look good. SMHMR gets 
    noisy at high masses
    ========================================

    '''
    if downsampled is None: 
        raise ValueError('the whole point of this function is to test downsampling...')

    # load in generic theta (parameter values)
    theta = Evol.defaultTheta(sfh) 

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
    subcat_down = eev_down.SH_catalog

    pretty_colors = prettycolors() 
    if test == 'sf_smf': # stellar mass function of SF population 
        isSF = np.where((subcat['gclass'] == 'star-forming') & (subcat['weights'] > 0.))
        isSF_down = np.where((subcat_down['gclass'] == 'star-forming') & (subcat_down['weights'] > 0.))

        fig = plt.figure(figsize=(7,7))
        sub = fig.add_subplot(111)

        snaps = [] # pick a handful of snapshots 
        for ii in range(2, nsnap0+1): 
            if (ii-1)%5 == 0: 
                snaps.append(ii)
        snaps.append(nsnap0) 

        for n in snaps[::-1]: # SMF of SF population at select snapshots 
            # full-sample
            smf_sf = Obvs.getMF(
                    subcat['snapshot'+str(n)+'_m.star'][isSF], 
                    weights=subcat['weights'][isSF])

            sub.plot(smf_sf[0], smf_sf[1], lw=2, c='b', alpha=0.05 * (21. - n))        
            
            # down-sample
            smf_sf = Obvs.getMF(
                    subcat_down['snapshot'+str(n)+'_m.star'][isSF_down], 
                    weights=subcat_down['weights'][isSF_down])

            sub.plot(smf_sf[0], smf_sf[1], lw=2, c='k', alpha=0.05 * (21. - n))        


        smf_sf = Obvs.getMF(subcat['m.star'][isSF], weights=subcat['weights'][isSF])
        sub.plot(smf_sf[0], smf_sf[1], lw=3, c='b', ls='-', label='Integrated')
        
        smf_sf = Obvs.getMF(subcat_down['m.star'][isSF_down], weights=subcat_down['weights'][isSF_down])
        sub.plot(smf_sf[0], smf_sf[1], lw=3, c='k', ls='-')

        sub.set_xlim([6., 12.])
        sub.set_xlabel('Stellar Masses $(\mathcal{M}_*)$', fontsize=25)
        sub.set_ylim([1e-6, 10**-1.75])
        sub.set_yscale('log')
        sub.set_ylabel('$\Phi$', fontsize=25)
        sub.legend(loc='upper right') 
        plt.show()

    elif test == 'smf': # stellar mass function of all galaxies 
        isSF = np.where(subcat['gclass'] == 'star-forming')
        isnotSF = np.where(subcat['gclass'] != 'star-forming')
        isSF_down = np.where(subcat_down['gclass'] == 'star-forming')
        isnotSF_down = np.where(subcat_down['gclass'] != 'star-forming')

        fig = plt.figure(figsize=(7,7))
        sub = fig.add_subplot(111)

        snaps = [] # pick a handful of snapshots 
        for ii in range(2, nsnap0+1): 
            if (ii-1)%7 == 0: 
                snaps.append(ii)
        snaps.append(nsnap0) 

        for n in snaps[::-1]: 
            # SHAM SMF 
            smf = Obvs.getMF(subcat['snapshot'+str(n)+'_m.sham'])
            sub.plot(smf[0], smf[1], lw=2, c='k', ls=':', alpha=0.05 * (21. - n))        

            # full-sample
            m_all = np.concatenate([subcat['snapshot'+str(n)+'_m.star'][isSF], subcat['snapshot'+str(n)+'_m.sham'][isnotSF]])
            w_all = np.concatenate([subcat['weights'][isSF], subcat['weights'][isnotSF]]) 

            smf = Obvs.getMF(m_all, weights=w_all)
            sub.plot(smf[0], smf[1], lw=2, c='b', alpha=0.05 * (21. - n))        
            
            # down-sample
            m_all = np.concatenate([
                subcat_down['snapshot'+str(n)+'_m.star'][isSF_down], 
                subcat_down['snapshot'+str(n)+'_m.sham'][isnotSF_down]])
            w_all = np.concatenate([
                subcat_down['weights'][isSF_down], 
                subcat_down['weights'][isnotSF_down]]) 
            smf = Obvs.getMF(m_all, weights=w_all)

            sub.plot(smf[0], smf[1], lw=2, c='g', alpha=0.05 * (21. - n))        

        # at snapshot 1  
        smf = Obvs.getMF(subcat['m.sham'])
        sub.plot(smf[0], smf[1], lw=2, c='k', ls=':')        

        m_all = np.concatenate([subcat['m.star'][isSF], subcat['m.sham'][isnotSF]])
        w_all = np.concatenate([subcat['weights'][isSF], subcat['weights'][isnotSF]]) 

        smf = Obvs.getMF(m_all, weights=w_all)
        sub.plot(smf[0], smf[1], lw=3, c='b', ls='-', label='Integrated')
        
        m_all = np.concatenate([
            subcat_down['m.star'][isSF_down], 
            subcat_down['m.sham'][isnotSF_down]])
        w_all = np.concatenate([
            subcat_down['weights'][isSF_down], 
            subcat_down['weights'][isnotSF_down]]) 
        smf = Obvs.getMF(m_all, weights=w_all)
        sub.plot(smf[0], smf[1], lw=3, c='g')

        sub.set_xlim([6., 12.])
        sub.set_xlabel('Stellar Masses $(\mathcal{M}_*)$', fontsize=25)
        sub.set_ylim([1e-6, 10**-1.75])
        sub.set_yscale('log')
        sub.set_ylabel('$\Phi$', fontsize=25)
        sub.legend(loc='upper right') 
        plt.show()

    elif test == 'pssfr': # compare the full and down - sampled P(SSFR)s
        obv_ssfr = Obvs.Ssfr()
        
        #  P(ssfr) at nsnap0  
        ssfr_bin_mids, ssfr_dists0 = obv_ssfr.Calculate(subcat['m.star0'], 
                subcat['sfr0'] - subcat['m.star0'], 
                subcat['weights'])
        
        # full sample 
        ssfr_bin_mids, pssfrs = obv_ssfr.Calculate(subcat['m.star'], 
                subcat['sfr'] - subcat['m.star'], 
                subcat['weights'])
        x_ssfrs = obv_ssfr.ssfr_bin_edges
        
        # down sample 
        ssfr_bin_mids, pssfrs_down = obv_ssfr.Calculate(subcat_down['m.star'], 
                subcat_down['sfr'] - subcat_down['m.star'], 
                subcat_down['weights'])
        x_ssfrs_down = obv_ssfr.ssfr_bin_edges

        fig = plt.figure(figsize=(20, 5))
        bkgd = fig.add_subplot(111, frameon=False)

        panel_mass_bins = [[9.7, 10.1], [10.1, 10.5], [10.5, 10.9], [10.9, 11.3]]
        for i_m, mass_bin in enumerate(panel_mass_bins): 
            sub = fig.add_subplot(1, 4, i_m+1)
            
            sub.plot(ssfr_bin_mids[i_m], ssfr_dists0[i_m], 
                    lw=3, ls='--', c='b', alpha=0.25)
            xx, yy = UT.bar_plot(x_ssfrs[i_m], pssfrs[i_m])
            sub.plot(xx, yy, lw=2, ls='-', c='k')
            xx, yy = UT.bar_plot(x_ssfrs_down[i_m], pssfrs_down[i_m])
            sub.plot(xx, yy, lw=3, ls='--', c='k')

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
        isSF_down = np.where(subcat_down['gclass'] == 'star-forming')

        smhmr = Obvs.Smhmr()

        fig = plt.figure()
        sub = fig.add_subplot(111)
        
        m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat['m.star'][isSF], subcat['halo.m'][isSF])
        sub.fill_between(m_mid, mu_mhalo - sig_mhalo, mu_mhalo + sig_mhalo, color='k', alpha=0.25, linewidth=0, edgecolor=None)
        print cnts[-10:] 
        m_mid, mu_mhalo, sig_mhalo, cnts = smhmr.Calculate(subcat_down['m.star'][isSF_down], subcat_down['halo.m'][isSF_down], 
                weights=subcat_down['weights'][isSF_down])
        sub.fill_between(m_mid, mu_mhalo - sig_mhalo, mu_mhalo + sig_mhalo, color='b', alpha=0.25, linewidth=0, edgecolor=None)
        print cnts[-10:] 
    
        sub.set_xlim([8., 12.])
        sub.set_xlabel('Stellar Mass $(\mathcal{M}_*)$', fontsize=25)
        sub.set_ylim([10., 15.])
        sub.set_ylabel('Halo Mass $(\mathcal{M}_{halo})$', fontsize=25)
        
        plt.show()

    elif test == 'sfms': 
        fig = plt.figure(figsize=(7,7))
        sub = fig.add_subplot(111)

        DFM.hist2d(subcat['m.star'], subcat['sfr'], weights=subcat['weights'], 
                levels=[0.68, 0.95], range=[[6., 12.], [-4., 2.]], color='#1F77B4', 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 

        DFM.hist2d(subcat_down['m.star'], subcat_down['sfr'], weights=subcat_down['weights'], 
                levels=[0.68, 0.95], range=[[6., 12.], [-4., 2.]], color='#FF7F0E', 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 

        sub.set_xlim([6., 12.])
        sub.set_xlabel('$\mathtt{log\;M_*}$', fontsize=25)
        sub.set_ylim([-4., 2.])
        sub.set_ylabel('$\mathtt{log\;SFR}$', fontsize=25)

        plt.show()

    return None


if __name__=="__main__": 
    #test_EvolverEvolve_downsample('smhmr', nsnap0=15, downsampled='14')
    test_EvolverEvolve_downsample('smf', nsnap0=15, downsampled='14', sfh='random_step_abias2')
    #for i in [15, 10, 5, 1]: 
    #    test_EvolverInitiate_downsample('sfms', i, nsnap0=15, downsampled='14')
