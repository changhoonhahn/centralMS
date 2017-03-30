import numpy as np 

import catalog as Cat
import evolver as Evol
import observables as Obvs
import util as UT

import matplotlib.pyplot as plt 
import bovy_plot as bovy
from ChangTools.plotting import prettycolors


def test_EvolverInitiate(test, nsnap): 
    ''' Tests for Initiate method in Evolver
    '''
    # load in Subhalo Catalog (pure centrals)
    subhist = Cat.PureCentralHistory(nsnap_ancestor=20)
    subcat = subhist.Read()#downsampled='33')

    # load in generic theta (parameter values)
    theta = Evol.defaultTheta() 

    eev = Evol.Evolver(subcat, theta, nsnap0=20)
    eev.Initiate()

    if test ==  'pssfr': # calculate P(SSFR) 
        obv_ssfr = Obvs.Ssfr()
        
        started = np.where(subcat['nsnap_start'] == nsnap)

        ssfr_bin_mids, ssfr_dists = obv_ssfr.Calculate(subcat['m.star0'][started], 
                subcat['sfr0'][started]-subcat['m.star0'][started], 
                subcat['weights'][started])

        fig = plt.figure(figsize=(20, 5))
        bkgd = fig.add_subplot(111, frameon=False)

        panel_mass_bins = [[9.7, 10.1], [10.1, 10.5], [10.5, 10.9], [10.9, 11.3]]
        for i_m, mass_bin in enumerate(panel_mass_bins): 
            sub = fig.add_subplot(1, 4, i_m+1)

            sub.plot(ssfr_bin_mids[i_m], ssfr_dists[i_m], 
                    lw=3, ls='-', c='k')
            
            # mark the SSFR of SFMS and Quiescent peak 
            sub.vlines(Obvs.SSFR_SFMS(0.5 * np.sum(mass_bin), UT.z_nsnap(20), theta_SFMS=theta['sfms']), 0., 1.7, 
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

    elif test == 'fq': # calculate quiescent fraction 
        obv_fq = Obvs.Fq()

        pretty_colors = prettycolors()

        fig = plt.figure(figsize=(6,6))
        sub = fig.add_subplot(111)

        started = np.where(subcat['nsnap_start'] == nsnap)
        print len(started[0]), ' galaxies'
        print np.sum(subcat['weights'][started])

        m_mid, fq, counts = obv_fq.Calculate(
                mass=subcat['m.star0'][started], 
                sfr=subcat['sfr0'][started], 
                z=UT.z_nsnap(nsnap), weights= subcat['weights'][started], theta_SFMS=theta['sfms'], counts=True)
        cc = pretty_colors[nsnap]
        sub.scatter(m_mid, fq, c=cc, s=10)
        sub.plot(m_mid, fq, c=cc)
        sub.plot(m_mid, obv_fq.model(m_mid, UT.z_nsnap(nsnap), lit='cosmos_tinker'), c=cc, ls='--')
        
        for i in range(len(m_mid)): 
            sub.text(m_mid[i], 0.05+fq[i], str(counts[i]))

        plt.show()
    
    elif test == 'smf_evol': # check the SMF evolution of the SF population
        fig = plt.figure(figsize=(7,7))
        sub = fig.add_subplot(111)

        for n in range(2, 21)[::-1]: 
            # identify SF population at snapshot
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
                    (subcat['gclass'] == 'star-forming') & 
                    (subcat['weights'] > 0.)
                    )
        smf_sf = Obvs.getMF(
                subcat['m.sham'][pop_sf], 
                weights=subcat['weights'][pop_sf])
        sub.plot(smf_sf[0], smf_sf[1], lw=3, c='k', ls='--', label='Snapshot 1')

        sub.set_xlim([6., 12.])
        sub.set_xlabel('Stellar Masses $(\mathcal{M}_*)$', fontsize=25)
        sub.set_ylim([1e-5, 10**-1.5])
        sub.set_yscale('log')
        sub.set_ylabel('$\Phi$', fontsize=25)
        sub.legend(loc='upper right') 
        plt.show()
    
    elif test == 'smf_M0': # check the SMF of galaxies with M_sham(z0) = 0
        fig = plt.figure(figsize=(7,7))
        sub = fig.add_subplot(111)

        blank = np.where((subcat['snapshot20_m.sham'] == 0.) & (subcat['weights'] > 0.))
            
        smf_blank = Obvs.getMF(subcat['m.sham'][blank], weights=subcat['weights'][blank])
        smf_tot = Obvs.getMF(subcat['m.sham'], weights=subcat['weights'])
        sub.plot(smf_tot[0], smf_blank[1]/smf_tot[1], lw=3, c='k', ls='-')
        
        sub.set_xlim([6., 12.])
        sub.set_xlabel('Stellar Masses $(\mathcal{M}_*)$', fontsize=25)
        #sub.set_ylim([1e-5, 10**-1.5])
        #sub.set_yscale('log')
        #sub.set_ylabel('$\Phi$', fontsize=25)
        plt.show()

    return None


def test_EvolverEvolve(test): 
    ''' Tests for Initiate method in Evolver
    '''
    # load in Subhalo Catalog (pure centrals)
    subhist = Cat.PureCentralHistory(nsnap_ancestor=20)
    subcat = subhist.Read()

    # load in generic theta (parameter values)
    theta = Evol.defaultTheta() 

    eev = Evol.Evolver(subcat, theta, nsnap0=20)
    eev.Initiate()

    eev.Evolve() 

    subcat = eev.SH_catalog
    #print subcat['m.sham'][np.where(subcat['snapshot20_m.sham'] == 0.)].max() 
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

    elif test == 'delMstar': 
        isSF = np.where(subcat['gclass'] == 'star-forming')

        delMstar = subcat['m.star'][isSF] - subcat['m.sham'][isSF]  # Delta M*

        bovy.scatterplot(subcat['m.star'][isSF], delMstar, scatter=True, s=2, 
                xrange=[8., 12.], yrange=[-4., 4.], xlabel='\mathtt{log\;M_*}', ylabel='\mathtt{log\;M_* - log\;M_{sham}}')

        plt.show()

    return None


def test_assignSFRs(): 
    ''' Test that Evol.assignSFRs function is working as expected. 
    ''' 
    zsnaps, tsnaps = UT.zt_table() # load in snapshot redshifts  

    subhist = Cat.PureCentralHistory(nsnap_ancestor=20)
    subcat = subhist.Read()
   
    # load in generic theta (parameter values)
    theta = Evol.defaultTheta() 

    out = Evol.assignSFRs(subcat['m.star'], np.repeat(zsnaps[20], len(subcat['m.star'])),
            theta_GV=theta['gv'], 
            theta_SFMS=theta['sfms'], 
            theta_FQ=theta['fq']) 
    # calculate their SSFRs
    obv_ssfr = Obvs.Ssfr()
    ssfr_bin_mids, ssfr_dists = obv_ssfr.Calculate(subcat['m.star'], out['SFR']-subcat['m.star'])

    fig = plt.figure(figsize=(20, 5))
    bkgd = fig.add_subplot(111, frameon=False)

    panel_mass_bins = [[9.7, 10.1], [10.1, 10.5], [10.5, 10.9], [10.9, 11.3]]
    for i_m, mass_bin in enumerate(panel_mass_bins): 
        sub = fig.add_subplot(1, 4, i_m+1)

        sub.plot(ssfr_bin_mids[i_m], ssfr_dists[i_m], 
                lw=3, ls='-', c='k')
        
        # mark the SSFR of SFMS and Quiescent peak 
        sub.vlines(Obvs.SSFR_SFMS(0.5 * np.sum(mass_bin), zsnaps[20], theta_SFMS=theta['sfms']), 0., 1.7, 
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
    return None


if __name__=="__main__": 
    test_EvolverEvolve('delMstar')
    #test_EvolverInitiate('pssfr', 15)
    #test_assignSFRs() 

