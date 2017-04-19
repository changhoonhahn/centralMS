import numpy as np 

import catalog as Cat
import evolver as Evol
import observables as Obvs
import util as UT

import matplotlib.pyplot as plt 
import bovy_plot as bovy
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors

def test_RandomStep_timescale(): 
    ''' Test the impact of the timescale for random step SFH scheme
    '''
    # load in Subhalo Catalog (pure centrals)
    subhist = Cat.PureCentralHistory(nsnap_ancestor=20)
    subcat = subhist.Read()

    # load in generic theta (parameter values)
    theta = Evol.defaultTheta('random_step') 
    
    for tstep in [0.1, 0.5, 1., 5.]: 
        theta['sfh'] = {'name': 'random_step', 
                'dt_min': tstep, 'dt_max': tstep, 'sigma': 0.3}

        eev = Evol.Evolver(subcat, theta, nsnap0=20)
        eev.Initiate()

        eev.Evolve() 

        subcat = eev.SH_catalog

        prettyplot() 
        pretty_colors = prettycolors() 

        isSF = np.where(subcat['gclass'] == 'star-forming')[0]
            
        fig = plt.figure(figsize=(25,7))
        sub = fig.add_subplot(1,3,1)

        for n in range(2, 21)[::-1]: 
            # identify SF population at snapshot
            smf_sf = Obvs.getMF(subcat['snapshot'+str(n)+'_m.star'][isSF], 
                    weights=subcat['weights'][isSF])

            sub.plot(smf_sf[0], smf_sf[1], lw=2, c='b', alpha=0.05 * (21. - n))        

        smf_sf = Obvs.getMF(subcat['m.star'][isSF], weights=subcat['weights'][isSF])
        sub.plot(smf_sf[0], smf_sf[1], lw=3, c='b', ls='-', label='Integrated')

        smf_sf_msham = Obvs.getMF(subcat['m.sham'][isSF], weights=subcat['weights'][isSF])
        sub.plot(smf_sf_msham[0], smf_sf_msham[1], lw=3, c='k', ls='--', label='SHAM')

        sub.set_xlim([6.75, 12.])
        sub.set_xlabel('Stellar Masses $(\mathcal{M}_*)$', fontsize=25)
        sub.set_ylim([1e-5, 10**-1.75])
        sub.set_yscale('log')
        sub.set_ylabel('log $\Phi$', fontsize=25)
        sub.legend(loc='upper right') 
            
        sub = fig.add_subplot(1,3,2)
        smhmr = Obvs.Smhmr()
        m_mid, mu_mstar, sig_mstar, cnts = smhmr.Calculate(subcat['halo.m'][isSF], subcat['m.star'][isSF])
        
        sub.errorbar(m_mid, mu_mstar, yerr=sig_mstar)
        sub.fill_between(m_mid, mu_mstar - 0.2, mu_mstar + 0.2, color='k', alpha=0.25, linewidth=0, edgecolor=None)

        sub.set_xlim([10.5, 14.])
        sub.set_xlabel('Halo Mass $(\mathcal{M}_{halo})$', fontsize=25)
        sub.set_ylim([8., 12.])
        sub.set_ylabel('Stellar Mass $(\mathcal{M}_*)$', fontsize=25)
            
        isSF = np.where((subcat['gclass'] == 'star-forming') & (subcat['nsnap_start'] == 20))[0]

        m_bin = np.arange(9.0, 12.5, 0.5)  
        i_bin = np.digitize(subcat['m.star0'][isSF], m_bin)
        
        sub = fig.add_subplot(1,3,3)
        for i in np.unique(i_bin): 
            i_rand = np.random.choice(np.where(i_bin == i)[0], size=1)[0]
            
            dsfrs = [subcat['sfr0'][isSF[i_rand]] - (Obvs.SSFR_SFMS(
                subcat['m.star0'][isSF[i_rand]], UT.z_nsnap(20), 
                theta_SFMS=eev.theta_sfms) + subcat['m.star0'][isSF[i_rand]])[0]]

            sub.text(UT.t_nsnap(20 - i) + 0.1, dsfrs[0] + 0.02, '$\mathcal{M}_* \sim $'+str(m_bin[i]), fontsize=15)

            for nn in range(2, 20)[::-1]: 
                M_nn = subcat['snapshot'+str(nn)+'_m.star'][isSF[i_rand]]
                mu_sfr = Obvs.SSFR_SFMS(M_nn, UT.z_nsnap(nn), theta_SFMS=eev.theta_sfms) + M_nn
                dsfrs.append(subcat['snapshot'+str(nn)+'_sfr'][isSF[i_rand]] - mu_sfr[0])

            mu_sfr = Obvs.SSFR_SFMS(subcat['m.star'][isSF[i_rand]], 
                    UT.z_nsnap(1), theta_SFMS=eev.theta_sfms) + subcat['m.star'][isSF[i_rand]]
            dsfrs.append(subcat['sfr'][isSF[i_rand]] - mu_sfr[0]) 
            sub.plot(UT.t_nsnap(range(1,21)[::-1]), dsfrs, c=pretty_colors[i], lw=2)

        sub.plot([UT.t_nsnap(20), UT.t_nsnap(1)], [0.3, 0.3], c='k', ls='--', lw=2)
        sub.plot([UT.t_nsnap(20), UT.t_nsnap(1)], [-0.3, -0.3], c='k', ls='--', lw=2)
        sub.set_xlim([UT.t_nsnap(20), UT.t_nsnap(1)])
        sub.set_xlabel('$t_{cosmic}$', fontsize=25)
        sub.set_ylim([-1., 1.])
        sub.set_ylabel('$\Delta$log SFR', fontsize=25)
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        fig.savefig(''.join([UT.fig_dir(), 'random_step.tstep', str(tstep), '.png']), bbox_inches='tight')
        plt.close() 
    return None


def EvolverPlots(sfh): 
    '''
    '''
    # load in Subhalo Catalog (pure centrals)
    subhist = Cat.PureCentralHistory(nsnap_ancestor=20)
    subcat = subhist.Read()

    # load in generic theta (parameter values)
    theta = Evol.defaultTheta(sfh) 

    eev = Evol.Evolver(subcat, theta, nsnap0=20)
    eev.Initiate()

    eev.Evolve() 

    subcat = eev.SH_catalog

    prettyplot() 
    pretty_colors = prettycolors() 

    isSF = np.where(subcat['gclass'] == 'star-forming')[0]
        
    fig = plt.figure(figsize=(25,7))
    sub = fig.add_subplot(1,3,1)

    for n in range(2, 21)[::-1]: 
        # identify SF population at snapshot
        smf_sf = Obvs.getMF(subcat['snapshot'+str(n)+'_m.star'][isSF], 
                weights=subcat['weights'][isSF])

        sub.plot(smf_sf[0], smf_sf[1], lw=2, c='b', alpha=0.05 * (21. - n))        

    smf_sf = Obvs.getMF(subcat['m.star'][isSF], weights=subcat['weights'][isSF])
    sub.plot(smf_sf[0], smf_sf[1], lw=3, c='b', ls='-', label='Integrated')

    smf_sf_msham = Obvs.getMF(subcat['m.sham'][isSF], weights=subcat['weights'][isSF])
    sub.plot(smf_sf_msham[0], smf_sf_msham[1], lw=3, c='k', ls='--', label='SHAM')

    sub.set_xlim([6.75, 12.])
    sub.set_xlabel('Stellar Masses $(\mathcal{M}_*)$', fontsize=25)
    sub.set_ylim([1e-5, 10**-1.75])
    sub.set_yscale('log')
    sub.set_ylabel('log $\Phi$', fontsize=25)
    sub.legend(loc='upper right') 
        
    sub = fig.add_subplot(1,3,2)
    smhmr = Obvs.Smhmr()
    m_mid, mu_mstar, sig_mstar, cnts = smhmr.Calculate(subcat['halo.m'][isSF], subcat['m.star'][isSF])
    
    sub.errorbar(m_mid, mu_mstar, yerr=sig_mstar)
    sub.fill_between(m_mid, mu_mstar - 0.2, mu_mstar + 0.2, color='k', alpha=0.25, linewidth=0, edgecolor=None)

    sub.set_xlim([10.5, 14.])
    sub.set_xlabel('Halo Mass $(\mathcal{M}_{halo})$', fontsize=25)
    sub.set_ylim([8., 12.])
    sub.set_ylabel('Stellar Mass $(\mathcal{M}_*)$', fontsize=25)
        
    isSF = np.where((subcat['gclass'] == 'star-forming') & (subcat['nsnap_start'] == 20))[0]

    m_bin = np.arange(9.0, 12.5, 0.5)  
    i_bin = np.digitize(subcat['m.star0'][isSF], m_bin)
    
    sub = fig.add_subplot(1,3,3)
    for i in np.unique(i_bin): 
        i_rand = np.random.choice(np.where(i_bin == i)[0], size=1)[0]
        
        dsfrs = [subcat['sfr0'][isSF[i_rand]] - (Obvs.SSFR_SFMS(
            subcat['m.star0'][isSF[i_rand]], UT.z_nsnap(20), 
            theta_SFMS=eev.theta_sfms) + subcat['m.star0'][isSF[i_rand]])[0]]

        sub.text(UT.t_nsnap(20 - i) + 0.1, dsfrs[0] + 0.02, '$\mathcal{M}_* \sim $'+str(m_bin[i]), fontsize=15)

        for nn in range(2, 20)[::-1]: 
            M_nn = subcat['snapshot'+str(nn)+'_m.star'][isSF[i_rand]]
            mu_sfr = Obvs.SSFR_SFMS(M_nn, UT.z_nsnap(nn), theta_SFMS=eev.theta_sfms) + M_nn
            dsfrs.append(subcat['snapshot'+str(nn)+'_sfr'][isSF[i_rand]] - mu_sfr[0])

        mu_sfr = Obvs.SSFR_SFMS(subcat['m.star'][isSF[i_rand]], 
                UT.z_nsnap(1), theta_SFMS=eev.theta_sfms) + subcat['m.star'][isSF[i_rand]]
        dsfrs.append(subcat['sfr'][isSF[i_rand]] - mu_sfr[0]) 
        sub.plot(UT.t_nsnap(range(1,21)[::-1]), dsfrs, c=pretty_colors[i], lw=2)

    sub.plot([UT.t_nsnap(20), UT.t_nsnap(1)], [0.3, 0.3], c='k', ls='--', lw=2)
    sub.plot([UT.t_nsnap(20), UT.t_nsnap(1)], [-0.3, -0.3], c='k', ls='--', lw=2)
    sub.set_xlim([UT.t_nsnap(20), UT.t_nsnap(1)])
    sub.set_xlabel('$t_{cosmic}$', fontsize=25)
    sub.set_ylim([-1., 1.])
    sub.set_ylabel('$\Delta$log SFR', fontsize=25)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.savefig(''.join([UT.fig_dir(), sfh+'_eval.png']), bbox_inches='tight')
    plt.close() 
    return None


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
    test_RandomStep_timescale()
    #EvolverPlots('constant_offset')
    #EvolverPlots('corr_constant_offset')
    #EvolverPlots('random_step')
    #test_EvolverEvolve('smhmr')
    #test_EvolverInitiate('pssfr', 15)
    #test_assignSFRs() 

