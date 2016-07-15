'''




'''
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors 

# --- local --- 
import util as UT 
import sfrs as SFR
import bovy_plot as bovy
import centralms as CMS 
import observables as Obvs

def plotCMS_SMF(evol_dict=None): 
    ''' Plot the stellar mass function of the integrated SFR population 
    and compare it to the theoretic stellar mass function
    '''
    # calculate SMF 
    cq = CMS.CentralQuenched()  # quenched population
    cq._Read_CenQue()
    # import evolved galaxy population 
    MSpop = CMS.EvolvedGalPop(cenque='default', evol_dict=evol_dict) 
    MSpop.Read() 

    smf = Obvs.getSMF(
            np.array(list(MSpop.mass)+list(cq.mass))
            )
    prettyplot() 
    pretty_colors = prettycolors() 

    # analytic SMF 
    theory_smf = Obvs.getSMF(
            np.array(list(MSpop.M_sham)+list(cq.M_sham))
            )
    #Obvs.analyticSMF(0.1, source='li-march')

    fig = plt.figure()
    sub = fig.add_subplot(111)

    sub.plot(theory_smf[0], theory_smf[1], lw=3, c='k', label='Theory')
    sub.plot(smf[0], smf[1], lw=3, c=pretty_colors[3], ls='--', label='Simul.')

    sub.set_ylim([10**-5, 10**-1])
    sub.set_xlim([6., 12.0])
    sub.set_yscale('log')
    sub.set_xlabel(r'Mass $\mathtt{M_*}$', fontsize=25) 
    sub.set_ylabel(r'Stellar Mass Function $\mathtt{\Phi}$', fontsize=25) 
    sub.legend(loc='upper right', frameon=False)
    
    fig_file = ''.join([UT.fig_dir(), 'SMF.CMS', MSpop._Spec_str(), '.png']) 
    fig.savefig(fig_file, bbox_inches='tight') 
    plt.close() 
    return None


def plotCMS_SMF_MS(evol_dict=None): 
    '''Plot the stellar mass function of the integrated SFR main sequence population 
    and compare it to the theoretic stellar mass function of the main sequence 
    population 
    '''
    # import evolved galaxy population 
    MSpop = CMS.EvolvedGalPop(cenque='default', evol_dict=evol_dict) 
    MSpop.Read() 
    MSonly = np.where(MSpop.t_quench == 999.)   # remove the quenching galaxies

    smf = Obvs.getSMF(MSpop.mass[MSonly])
    prettyplot() 
    pretty_colors = prettycolors() 

    # analytic SMF 
    theory_smf = Obvs.getSMF(MSpop.M_sham[MSonly])

    fig = plt.figure()
    sub = fig.add_subplot(111)

    sub.plot(theory_smf[0], theory_smf[1], lw=3, c='k', label='Theory')
    sub.plot(smf[0], smf[1], lw=3, c=pretty_colors[3], ls='--', label='MS Only')

    sub.set_ylim([10**-5, 10**-1])
    sub.set_xlim([6., 12.0])
    sub.set_yscale('log')
    sub.set_xlabel(r'Mass $\mathtt{M_*}$', fontsize=25) 
    sub.set_ylabel(r'Stellar Mass Function $\mathtt{\Phi}$', fontsize=25) 
    sub.legend(loc='upper right', frameon=False)
    
    fig_file = ''.join([UT.fig_dir(), 'SMF_MS.CMS', MSpop._Spec_str(), '.png']) 
    fig.savefig(fig_file, bbox_inches='tight') 
    plt.close() 
    return None


def plotCMS_SMF_comp(criteria='Mhalo0', population='all', Mtype='integ', evol_dict=None): 
    ''' Look at the composition of the SMF based on different criteria
    '''
    # import the catalogs
    cq = CMS.CentralQuenched()  # quenched population
    cq._Read_CenQue()
    
    MSpop = CMS.EvolvedGalPop(cenque='default', evol_dict=evol_dict) 
    MSpop.Read() 

    prettyplot() 
    pretty_colors = prettycolors() 
    fig = plt.figure()
    sub = fig.add_subplot(111)
    
    if criteria == 'Mhalo0': 
        keep_Q = np.where(cq.nsnap_genesis == 15) 
        keep_MS = np.where(MSpop.nsnap_genesis == 15) 
        if population == 'all': 
            if Mtype == 'integ': 
                masses = np.array(list(MSpop.mass[keep_MS]) + list(cq.mass[keep_Q]))
            elif Mtype == 'sham': 
                masses = np.array(list(MSpop.M_sham[keep_MS]) + list(cq.M_sham[keep_Q]))
            halomass_genesis = np.array(
                    list(MSpop.halomass_genesis[keep_MS]) + list(cq.halomass_genesis[keep_Q])
                    )
        elif population == 'ms': 
            if Mtype == 'integ': 
                masses = MSpop.mass[keep_MS]
            elif Mtype == 'sham': 
                masses = MSpop.M_sham[keep_MS]
            halomass_genesis = MSpop.halomass_genesis[keep_MS]

        halomass_bins = np.arange(halomass_genesis.min(), halomass_genesis.max(), 0.5) 
        for i in range(len(halomass_bins)-1): 
            bin = np.where(
                    (halomass_genesis >= halomass_bins[i]) & 
                    (halomass_genesis < halomass_bins[i+1]))

            smf = Obvs.getSMF(masses[bin], m_arr=None, dlogm=0.1, box=250, h=0.7)
            if i == 0: 
                smf_sum = np.zeros(len(smf[1]))
            sub.fill_between(smf[0], smf_sum,  smf_sum + smf[1], 
                    lw=0, color=pretty_colors[i], 
                    label=r"$\mathtt{M_{halo,0}} = ["+str(round(halomass_bins[i],1))+","+str(round(halomass_bins[i+1],1))+"$")
            smf_sum += smf[1]

    elif criteria == 't0': 
        if population == 'all': 
            if Mtype == 'integ': 
                masses = np.array(list(MSpop.mass) + list(cq.mass))
            elif Mtype == 'sham': 
                masses = np.array(list(MSpop.M_sham) + list(cq.M_sham))
            t_genesis = np.array(list(MSpop.tsnap_genesis) + list(cq.tsnap_genesis))
        elif population == 'ms': 
            if Mtype == 'integ': 
                masses = MSpop.mass
            elif Mtype == 'sham': 
                masses = MSpop.M_sham
            t_genesis = MSpop.tsnap_genesis

        for it, tt in enumerate(np.unique(t_genesis)): 
            t_bin = np.where(t_genesis == tt) 

            smf = Obvs.getSMF(masses[t_bin], m_arr=None, dlogm=0.1, box=250, h=0.7)
            if it == 0: 
                smf_sum = np.zeros(len(smf[1]))
            sub.fill_between(smf[0], smf_sum,  smf_sum + smf[1], 
                    lw=0, color=pretty_colors[it], 
                    label=r"$\mathtt{t_0} = "+str(round(tt,1))+"$")
            smf_sum += smf[1]

    sub.set_ylim([10**-5, 10**-1])
    sub.set_xlim([6., 12.0])
    sub.set_yscale('log')
    sub.set_xlabel(r'Mass $\mathtt{M_*}$', fontsize=25) 
    sub.set_ylabel(r'Stellar Mass Function $\mathtt{\Phi}$', fontsize=25) 
    sub.legend(loc='upper right', frameon=False)
    
    fig_file = ''.join([UT.fig_dir(), 
        'test_CentralMS_SMF', 
        '.', population, 
        '.criteria_', criteria, 
        '.M', Mtype, 
        '.sfh_', evol_dict['sfh']['name'], 
        '.mass_', evol_dict['mass']['type'], '_', str(evol_dict['mass']['t_step']), 
        '.png']) 
    fig.savefig(fig_file) 
    plt.close() 
    return None


def plotCMS_SFMS(evol_dict=None):
    ''' Plot the SFR-M* relation of the SFMS and compare to expectations
    '''
    # import evolved galaxy population 
    cms = CMS.CentralMS(cenque='default')
    cms._Read_CenQue()
    MSpop = CMS.EvolvedGalPop(cenque='default', evol_dict=evol_dict) 
    MSpop.Read() 
    MSonly = np.where(MSpop.t_quench == 999.) 
    
    delta_m = 0.5
    mbins = np.arange(MSpop.mass[MSonly].min(), MSpop.mass[MSonly].max(), delta_m) 
    
    sfr_percent = np.zeros((len(mbins)-1, 4))
    for im, mbin in enumerate(mbins[:-1]):  
        within = np.where(
                (MSpop.mass[MSonly] > mbin) & 
                (MSpop.mass[MSonly] <= mbin + delta_m) 
                ) 
        sfr_percent[im,:] = np.percentile(MSpop.sfr[MSonly[0][within]], [2.5, 16, 84, 97.5])
        
    prettyplot() 
    pretty_colors = prettycolors()
    fig = plt.figure()
    sub = fig.add_subplot(111)

    # parameterized observed SFMS 
    obvs_mu_sfr = SFR.AverageLogSFR_sfms(0.5*(mbins[:-1]+mbins[1:]), 0.05, sfms_dict=cms.sfms_dict)
    obvs_sig_sfr = SFR.ScatterLogSFR_sfms(0.5*(mbins[:-1]+mbins[1:]), 0.05, sfms_dict=cms.sfms_dict)
    sub.fill_between(0.5*(mbins[:-1] + mbins[1:]), 
            obvs_mu_sfr - 2.*obvs_sig_sfr, obvs_mu_sfr + 2.*obvs_sig_sfr, 
            color=pretty_colors[1], alpha=0.3, edgecolor=None, lw=0) 
    sub.fill_between(0.5*(mbins[:-1] + mbins[1:]), 
            obvs_mu_sfr - obvs_sig_sfr, obvs_mu_sfr + obvs_sig_sfr, 
            color=pretty_colors[1], alpha=0.5, edgecolor=None, lw=0, label='Observations') 

    # simulation 
    sub.fill_between(0.5*(mbins[:-1] + mbins[1:]), sfr_percent[:,0], sfr_percent[:,-1],  
            color=pretty_colors[3], alpha=0.3, edgecolor=None, lw=0) 
    sub.fill_between(0.5*(mbins[:-1] + mbins[1:]), sfr_percent[:,1], sfr_percent[:,-2], 
            color=pretty_colors[3], alpha=0.5, edgecolor=None, lw=0, label='Simulated') 

    # x,y axis
    sub.set_xlim([9.0, 12.0]) 
    sub.set_xlabel(r'$\mathtt{M_*}\; [\mathtt{M}_\odot]$', fontsize=25)
    sub.set_ylim([-2., 1.]) 
    sub.set_ylabel(r'$\mathtt{SFR}\; [\mathtt{M}_\odot/\mathtt{yr}]$', fontsize=25)

    sub.legend(loc='lower right', numpoints=1, markerscale=1.) 

    fig_file = ''.join([UT.fig_dir(), 'SFMS.CMS', MSpop._Spec_str(), '.png'])
    plt.savefig(fig_file, bbox_inches='tight') 
    plt.close()
    return None


def plotCMS_SFMS_Pssfr(evol_dict=None):
    ''' Plot SSFR distribution of the SFMS in mass bins. 
    The SFMS SSFR distirbution *should* be a Gaussian!
    '''
    # import evolved galaxy population 
    cms = CMS.CentralMS(cenque='default')
    cms._Read_CenQue()
    MSpop = CMS.EvolvedGalPop(cenque='default', evol_dict=evol_dict) 
    MSpop.Read() 

    mass_bins = [[9.7, 10.1], [10.1, 10.5], [10.5, 10.9], [10.9, 11.3]]
    
    prettyplot() 
    pretty_colors = prettycolors()
    fig = plt.figure(figsize=(8,8))
    sub = [fig.add_subplot(2,2, ii+1) for ii in xrange(len(mass_bins))]
    subsub = fig.add_subplot(111, frameon=False)
    
    for im, mbin in enumerate(mass_bins): 
        MSonly_bin = np.where(
                (MSpop.t_quench == 999.) &
                (MSpop.mass > mbin[0]) &
                (MSpop.mass <= mbin[1])
                )

        bin_ssfrs = MSpop.sfr[MSonly_bin] - MSpop.mass[MSonly_bin]
        # calculate P(SSFR)
        Pssfr, bin_edges = np.histogram(
                bin_ssfrs,
                range=[-13.0, -7.0],
                bins=40,
                normed=True)
    
        # P(SSFR) for the MS galaxies in the mass bin 
        sub[im].plot(0.5*(bin_edges[:-1] + bin_edges[1:]), Pssfr, 
                lw=4, c=pretty_colors[3]) 
        
        # Gaussian  
        mu_ssfr = SFR.AverageLogSFR_sfms(0.5*(mbin[0]+mbin[1]), 0.05, sfms_dict=cms.sfms_dict) -\
                0.5*(mbin[0]+mbin[1])
        sig_ssfr = SFR.ScatterLogSFR_sfms(0.5*(mbin[0]+mbin[1]), 0.05, sfms_dict=cms.sfms_dict)
        norm_Pssfr, bin_edges = np.histogram(
                mu_ssfr + sig_ssfr * np.random.randn(len(bin_ssfrs)),
                range=[-13.0, -7.0],
                bins=40,
                normed=True)
        sub[im].plot(0.5*(bin_edges[:-1] + bin_edges[1:]), norm_Pssfr, lw=4, ls='--', c='k') 

        massbin_str = ''.join([         # mark the mass bins
            r'$\mathtt{log \; M_{*} = [', 
            str(mbin[0]), ',\;', str(mbin[1]), ']}$' ])
        sub[im].text(-11.8, 1.8, massbin_str, fontsize=15)
        sub[im].set_xlim([-12.0, -9.0])
        sub[im].set_ylim([0.0, 2.0])
        if im in [0,1]: 
            sub[im].set_xticklabels([])
        if im in [1,3]: 
            sub[im].set_yticklabels([])
        if im in [2]: 
            sub[im].set_xticklabels([-12., '', -11., '', -10.])
            sub[im].set_yticklabels([0.0, 0.5, 1.0, 1.5])
        if im in [3]:
            sub[im].set_xticklabels([-12., '', -11., '', -10., '', -9.])

    subsub.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    subsub.set_xlabel(r'$\mathtt{log \; SSFR \;[yr^{-1}]}$', fontsize=25) 
    subsub.set_ylabel(r'$\mathtt{P(log \; SSFR)}$', fontsize=25) 
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    fig_file = ''.join([UT.fig_dir(), 'Pssfr_SFMS.CMS', MSpop._Spec_str(), '.png'])
    plt.savefig(fig_file, bbox_inches='tight') 
    plt.close()
    return None


def plotSHAM_SMHMR_MS(): 
    ''' Plot the Stellar Mass to Halo Mass Relation of the evolved Gal population 
    '''
    evol_dict = {
            'initial': {'assembly_bias': 'none'}, 
            'sfh': {'name': 'constant_offset'}, 
            'mass': {'type': 'euler', 'f_retain': 0.6, 't_step': 0.01} 
            } 
    # import evolved galaxy population 
    egp = CMS.EvolvedGalPop(cenque='default', evol_dict=evol_dict) 
    egp.Read() 
    
    # Calculate the SMHMR for the EvolvedGalPop
    m_halo_bin = np.arange(10., 15.25, 0.25)    # M_halo bins 
    m_halo_mid = 0.5 * (m_halo_bin[:-1] + m_halo_bin[1:]) # M_halo bin mid

    smhmr = np.zeros((len(m_halo_bin)-1, 5)) 
    for im, m_mid in enumerate(m_halo_mid): 
        inbin = np.where(
                (egp.halo_mass > m_halo_bin[im]) &
                (egp.halo_mass <= m_halo_bin[im+1])) 
        smhmr[im,:] = np.percentile(egp.M_sham[inbin], [2.5, 16, 50, 84, 97.5])

    # "observations" with observed scatter of 0.2 dex 
    obvs_smhmr = np.zeros((len(m_halo_bin)-1, 2))
    obvs_smhmr[:,0] = smhmr[:,2] - 0.2      
    obvs_smhmr[:,1] = smhmr[:,2] + 0.2      

    prettyplot() 
    pretty_colors = prettycolors() 
    fig = plt.figure()
    sub = fig.add_subplot(111)

    sub.fill_between(m_halo_mid, smhmr[:,0], smhmr[:,-1], 
        color=pretty_colors[1], alpha=0.25, edgecolor=None, lw=0) 
    sub.fill_between(m_halo_mid, smhmr[:,1], smhmr[:,-2], 
        color=pretty_colors[1], alpha=0.5, edgecolor=None, lw=0) 
    
    sub.plot(m_halo_mid, obvs_smhmr[:,0], 
        color='k', ls='--', lw=3) 
    sub.plot(m_halo_mid, obvs_smhmr[:,1], 
        color='k', ls='--', lw=3) 

    sub.set_xlim([10., 15.0])
    sub.set_ylim([9., 12.0])
    sub.set_xlabel(r'$\mathtt{log}\;\mathtt{M_{halo}}\;\;[\mathtt{M_\odot}]$', fontsize=25) 
    sub.set_ylabel(r'$\mathtt{log}\;\mathtt{M_{SHAM}}\;\;[\mathtt{M_\odot}]$', fontsize=25) 

    sub.legend(loc='upper right') 
    
    fig_file = ''.join([UT.fig_dir(), 'test_CentralMS_SMHMR_MS', '.M_sham', '.png']) 
    fig.savefig(fig_file, bbox_inches='tight') 
    plt.close() 
    return None


def plotCMS_SMHMR_MS(evol_dict=None): 
    ''' Plot the Stellar Mass to Halo Mass Relation of the evolved Gal population 
    '''
    # import evolved galaxy population 
    egp = CMS.EvolvedGalPop(cenque='default', evol_dict=evol_dict) 
    egp.Read() 
    
    # Calculate the SMHMR for the EvolvedGalPop
    m_halo_bin = np.arange(10., 15.25, 0.25)    # M_halo bins 
    m_halo_mid = 0.5 * (m_halo_bin[:-1] + m_halo_bin[1:]) # M_halo bin mid

    smhmr = np.zeros((len(m_halo_bin)-1, 5)) 
    for im, m_mid in enumerate(m_halo_mid): 
        inbin = np.where(
                (egp.halo_mass > m_halo_bin[im]) &
                (egp.halo_mass <= m_halo_bin[im+1])) 
        smhmr[im,:] = np.percentile(egp.mass[inbin], [2.5, 16, 50, 84, 97.5])

    # "observations" with observed scatter of 0.2 dex 
    obvs_smhmr = np.zeros((len(m_halo_bin)-1, 2))
    obvs_smhmr[:,0] = smhmr[:,2] - 0.2      
    obvs_smhmr[:,1] = smhmr[:,2] + 0.2      

    prettyplot() 
    pretty_colors = prettycolors() 
    fig = plt.figure()
    sub = fig.add_subplot(111)

    sub.fill_between(m_halo_mid, smhmr[:,0], smhmr[:,-1], 
        color=pretty_colors[1], alpha=0.25, edgecolor=None, lw=0) 
    sub.fill_between(m_halo_mid, smhmr[:,1], smhmr[:,-2], 
        color=pretty_colors[1], alpha=0.5, edgecolor=None, lw=0, 
        label=r'$\mathtt{Simulated}$') 
    
    sub.plot(m_halo_mid, obvs_smhmr[:,0], 
        color='k', ls='--', lw=3) 
    sub.plot(m_halo_mid, obvs_smhmr[:,1], 
        color='k', ls='--', lw=3, 
        label=r"$\sigma = \mathtt{0.2}$") 

    sub.set_xlim([10., 15.0])
    sub.set_ylim([9., 12.0])
    sub.set_xlabel(r'$\mathtt{log}\;\mathtt{M_{halo}}\;\;[\mathtt{M_\odot}]$', fontsize=25) 
    sub.set_ylabel(r'$\mathtt{log}\;\mathtt{M_{*}}\;\;[\mathtt{M_\odot}]$', fontsize=25) 

    sub.legend(loc='upper right') 
    
    fig_file = ''.join([UT.fig_dir(), 'SMHMR_MS.CMS', egp._Spec_str(), '.png']) 
    fig.savefig(fig_file, bbox_inches='tight') 
    plt.close() 
    return None


def plotCMS_SFH(evol_dict=None): 
    ''' Plot the star formation history of star forming main sequence galaxies
    '''
    # import evolved galaxy population 
    egp = CMS.EvolvedGalPop(cenque='default', evol_dict=evol_dict) 
    egp.Read() 

    MSonly = np.where(egp.t_quench == 999.)     # only keep main-sequence galaxies
    MSonly_rand = np.random.choice(MSonly[0], size=10)

    z_acc, t_acc = UT.zt_table()
    t_cosmic  = t_acc[1:16]
    dt = 0.1
    
    t_arr = np.arange(t_cosmic.min(), t_cosmic.max()+dt, dt)

    prettyplot() 
    pretty_colors = prettycolors() 
    fig = plt.figure()
    sub = fig.add_subplot(111)
    
    dsfrt = np.zeros((len(t_arr), len(MSonly_rand)))
    for i_t, tt in enumerate(t_arr): 
        dsfr = SFR.dSFR_MS(tt, egp.sfh_dict) 
        for ii, i_gal in enumerate(MSonly_rand): 
            dsfrt[i_t,ii] = dsfr[i_gal]
    
    for ii, i_gal in enumerate(MSonly_rand): 
        tlim = np.where(t_arr > egp.tsnap_genesis[i_gal]) 
        sub.plot(t_arr[tlim], dsfrt[:,ii][tlim], lw=3, c=pretty_colors[ii])

    sub.set_xlim([5.0, 13.5])
    sub.set_xlabel(r'$\mathtt{t_{cosmic}}$', fontsize=25)
    sub.set_ylim([-1., 1.])
    sub.set_ylabel(r'$\mathtt{SFR(t) - <SFR_{MS}(t)>}$', fontsize=25)

    fig_file = ''.join([UT.fig_dir(), 'SFH.CMS', egp._Spec_str(), '.png']) 
    fig.savefig(fig_file, bbox_inches='tight') 
    plt.close()




if __name__=='__main__': 
    for scat in [0.0, 0.1, 0.2, 0.3]: 
        evol_dict = {
                'initial': {'assembly_bias': 'longterm', 'scatter': scat}, 
                'sfh': {'name': 'random_step', 'sigma':0.3, 'dt_min': 0.1, 'dt_max':0.5}, 
                'mass': {'type': 'euler', 'f_retain': 0.6, 't_step': 0.01} 
                } 
        #'sfh': {'name': 'constant_offset'}, 
        #'sfh': {'name': 'random_step', 'sigma':0.3, 'dt_min': 0.1, 'dt_max':0.5}, 
        #plotCMS_SMF(evol_dict=evol_dict)
        #plotCMS_SMF_MS(evol_dict=evol_dict)
        #plotCMS_SFMS(evol_dict=evol_dict)
        #plotCMS_SFMS_Pssfr(evol_dict=evol_dict)
        #plotCMS_SMHMR_MS(evol_dict=evol_dict)
        plotCMS_SFH(evol_dict=evol_dict)
    #plotCMS_SMF_comp(criteria='t0', population='ms', evol_dict=evol_dict)
    #plotCMS_SMF_comp(criteria='t0', population='ms', Mtype='sham', evol_dict=evol_dict)
