'''



'''
import time
from sys import getsizeof
import numpy as np 
from scipy.special import erfinv
from scipy.interpolate import interp1d

# --- local --- 
import util as UT
import matplotlib.pyplot as plt


def logSFR_initiate(SHsnaps, indices, theta_sfh=None, theta_sfms=None):
    ''' initiate log SFR function for Evolver.Evolve() method
    '''
    mu_sfr0 = SFR_sfms(SHsnaps['m.star0'][indices], UT.z_nsnap(SHsnaps['nsnap_start'][indices]), theta_sfms)

    if theta_sfh['name'] == 'constant_offset':
        # constant d_logSFR 
        F_sfr = _logSFR_dSFR 
        sfr_kwargs = {'dSFR': SHsnaps['sfr0'][indices] - mu_sfr0,  # offset
                'theta_sfms': theta_sfms}

    elif theta_sfh['name'] == 'corr_constant_offset': 
        # constant d_logSFR (assigned based on halo accretion rate) 
        #dSFR0 = SHsnaps['sfr0'][indices] - mu_sfr

        # now rank order it based on halo accretion rate
        dMhalo = SHsnaps['halo.m'][indices] - SHsnaps['halo.m0'][indices]
        
        m_kind = SHsnaps[theta_sfh['m.kind']+'0'][indices] # how do we bin the SFRs?
        dm_kind = theta_sfh['dm.kind'] # bins to rank order
    
        # scatter from noise -- subtract intrinsic assembly bias scatter from sig_SFMS 
        sig_noise = np.sqrt(0.3**2 - theta_sfh['sig_abias']**2)

        # slow and inefficient 

        dsfr = np.repeat(-999., len(dMhalo))
        for nn in np.arange(1, 21): 
            started = np.where(SHsnaps['nsnap_start'][indices] == nn)[0]

            m_kind_bin = np.arange(m_kind[started].min(), m_kind[started].max()+dm_kind, dm_kind)
            ibin = np.digitize(m_kind[started], m_kind_bin) 
            for i in np.unique(ibin): 
                inbin = np.where(ibin == i)

                isort = np.argsort(dMhalo[started[inbin]])

                dsfr[started[inbin[0][isort]]] = \
                        np.sort(np.random.randn(len(inbin[0]))) * theta_sfh['sig_abias']

        assert dsfr.min() != -999.
        
        if sig_noise > 0.: 
            dsfr += sig_noise * np.random.randn(len(dMhalo))
        
        # correct initial SFR to match long-term assembly bias 
        SHsnaps['sfr0'][indices] = mu_sfr0 + dsfr

        F_sfr = _logSFR_dSFR 

        sfr_kwargs = {'dSFR': dsfr, 'theta_sfms': theta_sfms}

    elif theta_sfh['name'] == 'random_step': 
        # completely random 
        # amplitude is sampled randomly from a Gaussian with sig_logSFR = 0.3 
        # time steps are sampled randomly from a unifrom distribution [dt_min, dt_max]
        if 'dt_min' not in theta_sfh: 
            raise ValueError
        if 'dt_max' not in theta_sfh: 
            raise ValueError
                
        # Random step function duty cycle 
        del_t_max = UT.t_nsnap(1) - UT.t_nsnap(SHsnaps['nsnap0']) #'nsnap_start'][indices].max()) 
        
        # the range of the steps 
        tshift_min = theta_sfh['dt_min'] 
        tshift_max = theta_sfh['dt_max'] 

        # get the times when the amplitude changes 
        n_col = int(np.ceil(del_t_max/tshift_min))+1  # number of columns 
        n_gal = len(indices)

        tshift = np.zeros((n_gal, n_col))
        tshift[:,1:] = np.random.uniform(tshift_min, tshift_max, size=(n_gal, n_col-1))
        tsteps = np.cumsum(tshift , axis=1) + np.tile(UT.t_nsnap(SHsnaps['nsnap_start'][indices]), (n_col, 1)).T
        del tshift
        # make sure everything evolves properly until the end
        assert tsteps[range(n_gal), n_col-1].min() > UT.t_nsnap(1)

        dlogSFR_amp = np.random.randn(n_gal, n_col) * theta_sfh['sigma']
        dlogSFR_amp[:,0] = SHsnaps['sfr0'][indices] - mu_sfr0

        F_sfr = _logSFR_dSFR_tsteps
        
        sfr_kwargs = {'dlogSFR_amp': dlogSFR_amp, 'tsteps': tsteps,'theta_sfms': theta_sfms}
    
    elif theta_sfh['name'] == 'random_step_fluct': 
        # completely random amplitude that is sampled from a Gaussian with sig_logSFR = 0.3 
        # EXCEPT each adjacent timesteps have alternating sign amplitudes
        # time steps are sampled randomly from a unifrom distribution [dt_min, dt_max]
        if 'dt_min' not in theta_sfh: 
            raise ValueError
        if 'dt_max' not in theta_sfh: 
            raise ValueError
                
        # Random step function duty cycle 
        del_t_max = UT.t_nsnap(1) - UT.t_nsnap(SHsnaps['nsnap0']) #'nsnap_start'][indices].max()) 
        
        # the range of the steps 
        tshift_min = theta_sfh['dt_min'] 
        tshift_max = theta_sfh['dt_max'] 

        # get the times when the amplitude changes 
        n_col = int(np.ceil(del_t_max/tshift_min))+1  # number of columns 
        n_gal = len(indices)    # number of galaxies
        tshift = np.zeros((n_gal, n_col))
        tshift[:,1:] = np.random.uniform(tshift_min, tshift_max, size=(n_gal, n_col-1))
        tsteps = np.cumsum(tshift , axis=1) + np.tile(UT.t_nsnap(SHsnaps['nsnap_start'][indices]), (n_col, 1)).T
        del tshift
        # make sure everything evolves properly until the end
        assert tsteps[range(n_gal), n_col-1].min() > UT.t_nsnap(1)
        
        # all positive amplitudes
        dlogSFR_amp = np.abs(np.random.randn(n_gal, n_col)) * theta_sfh['sigma']
        # now make every other time step negative!
        plusminus = np.ones((n_gal, n_col))
        for i in range(np.int(np.ceil(np.float(n_col)/2.))): 
            plusminus[:, 2*i] *= -1. 
        dlogSFR_amp *= plusminus
        
        # make sure that nsnap0 is consistent with initial conditions!
        dlogSFR_amp[:,0] = SHsnaps['sfr0'][indices] - mu_sfr0

        F_sfr = _logSFR_dSFR_tsteps
        
        sfr_kwargs = {'dlogSFR_amp': dlogSFR_amp, 'tsteps': tsteps,'theta_sfms': theta_sfms}

    elif theta_sfh['name'] == 'random_step_most_abias': 
        # random steps with assembly bias where SFH is correlated with 
        # halo mass growth from snapshots nsnap0 to 1. This is designed to 
        # maximize assembly bias 
        if 'dt_min' not in theta_sfh: 
            raise ValueError
        if 'dt_max' not in theta_sfh: 
            raise ValueError
        if 'sigma_corr' not in theta_sfh:  
            raise ValueError
        else: 
            if theta_sfh['sigma_corr'] <= 0.: 
                raise ValueError("no assembly bias; dont use this SFH")
        # first calculate the time at which the SFH changes (random step function duty cycle)
        del_t_max = UT.t_nsnap(1) - UT.t_nsnap(SHsnaps['nsnap0'])#SHsnaps['nsnap_start'][indices].max()) 
        
        tshift_min = theta_sfh['dt_min'] # the range of the steps 
        tshift_max = theta_sfh['dt_max'] 

        # get the times when the amplitude changes 
        n_col = int(np.ceil(del_t_max/tshift_min))+1  # number of columns 
        n_gal = len(indices)

        tshift = np.zeros((n_gal, n_col), dtype=np.float32)
        if tshift_min == tshift_max: 
            tshift[:,1:] = np.repeat(tshift_min, n_gal * (n_col-1)).reshape((n_gal, n_col-1))
        else: 
            tshift[:,1:] = np.random.uniform(tshift_min, tshift_max, size=(n_gal, n_col-1))
        tsteps = np.cumsum(tshift , axis=1) + np.tile(UT.t_nsnap(SHsnaps['nsnap_start'][indices]), (n_col, 1)).T
        del tshift
    
        # M_halo initial 
        Mh_i = SHsnaps['halo.m0'][indices]
        # M_halo final 
        Mh_f = SHsnaps['halo.m'][indices]
        
        f_dMh = 1. - 10.**(Mh_i - Mh_f) #10.**(Mh_f - Mh_i - 10)
        #f_dMh = Mh_f - Mh_i
        
        # M_h of the galaxies throughout the snapshots 
        # Mh_snaps = ngal x nsnap0+3 matrix 
        Mh_snaps = np.zeros((n_gal, SHsnaps['nsnap0']))#, dtype=np.float32)
        Mh_snaps[:,0] =  SHsnaps['halo.m'][indices]
        for isnap in range(2, SHsnaps['nsnap0']+1): 
            Mh_snaps[:,isnap-1] = SHsnaps['halo.m.snap'+str(isnap)][indices]
        
        z_snaps = UT.z_nsnap(range(1, SHsnaps['nsnap0']+10))
        t_snaps = UT.t_nsnap(range(1, SHsnaps['nsnap0']+10))
    
        # M_h at the steps 
        Mh_steps = np.zeros(tsteps.shape, dtype=np.float32) 
        for i_g in range(n_gal): 
            insim = np.where(Mh_snaps[i_g,:] > 0.)[0] # some halos did not exist far back
            Mh_snaps_i = 10**(Mh_snaps[i_g,insim] - 10.) # scaled by 10^10 for ease
            t_snaps_i = t_snaps[insim]
            # SFH is correlated with halo growth dt_delay earlier  
            tstep_i = tsteps[i_g,:]
            # note that x has to be monotonically for np.interp 
            Mh_steps[i_g,:] = np.interp(tstep_i, t_snaps_i[::-1], Mh_snaps_i[::-1])

        # calculate the d(log SFR) amplitude at t_steps 
        dlogSFR_amp = np.zeros(tsteps.shape)#, dtype=np.float32)
        for ii in range(tsteps.shape[1]): 
            Mh_step = np.log10(Mh_steps[:,ii]) + 10.
            mh_bins = np.arange(Mh_step.min(), Mh_step.max()+0.2, 0.2)
            
            #fig = plt.figure(1, figsize=(8*np.int(np.ceil(np.float(len(mh_bins))/3.)+1.), 15))
            for i_m in range(len(mh_bins)-1): 
                inbin = np.where((Mh_step >= mh_bins[i_m]) & 
                        (Mh_step < mh_bins[i_m+1]))
                n_bin = len(inbin[0])
                 
                isort = np.argsort(-1. * f_dMh[inbin])

                irank = np.zeros(n_bin)
                irank[isort] = (np.arange(n_bin) + 0.5)/np.float(n_bin)
                
                # i_rank = 1/2 (1 - erf(x/sqrt(2)))
                # x = (SFR - avg_SFR)/sigma_SFR
                # dSFR = sigma_SFR * sqrt(2) * erfinv(1 - 2 i_rank)
                dlogSFR_amp[inbin, ii] = \
                        theta_sfh['sigma_corr'] * 1.41421356 * erfinv(1. - 2. * irank) 

                #sub = fig.add_subplot(3, np.int(np.ceil(np.float(len(mh_bins))/3.)+1), i_m+1)
                #sub.scatter(f_dMh[inbin], 0.3 * np.random.randn(n_bin), c='k')
                #sub.scatter(f_dMh[inbin], dlogSFR_amp[inbin, ii] + \
                #        np.sqrt(0.3**2 - theta_sfh['sigma_corr']**2) * np.random.randn(n_bin), c='r', lw=0)
                #sub.set_xlim([-1., 1.])
                #sub.set_ylim([-1., 1.])

            #plt.show()
            #raise ValueError
        del f_dMh
        del Mh_steps
        
        # add in intrinsic scatter
        dlogSFR_int = np.random.randn(n_gal, n_col) * np.sqrt(theta_sfh['sigma_tot']**2 - theta_sfh['sigma_corr']**2) 
        dlogSFR_amp += dlogSFR_int

        #dlogSFR_amp[:,0] = SHsnaps['sfr0'][indices] - mu_sfr0
        SHsnaps['sfr0'][indices] = mu_sfr0 + dlogSFR_amp[:,0]
        
        F_sfr = _logSFR_dSFR_tsteps
        
        sfr_kwargs = {'dlogSFR_amp': dlogSFR_amp, 'tsteps': tsteps,'theta_sfms': theta_sfms}

    elif theta_sfh['name'] == 'random_step_abias': 
        # random steps with assembly bias  
        if 'dt_min' not in theta_sfh: 
            raise ValueError
        if 'dt_max' not in theta_sfh: 
            raise ValueError

        # Random step function duty cycle 
        del_t_max = UT.t_nsnap(1) - UT.t_nsnap(SHsnaps['nsnap0'])#SHsnaps['nsnap_start'][indices].max()) 
        
        # the range of the steps 
        tshift_min = theta_sfh['dt_min'] 
        tshift_max = theta_sfh['dt_max'] 

        # get the times when the amplitude changes 
        n_col = int(np.ceil(del_t_max/tshift_min))+1  # number of columns 
        n_gal = len(indices)

        tshift = np.zeros((n_gal, n_col))
        tshift[:,1:] = np.random.uniform(tshift_min, tshift_max, size=(n_gal, n_col-1))
        tsteps = np.cumsum(tshift , axis=1) + np.tile(UT.t_nsnap(SHsnaps['nsnap_start'][indices]), (n_col, 1)).T
        del tshift
        
        assert tsteps[range(n_gal), n_col-1].min() > UT.t_nsnap(1)
        
        # calculate d(logSFR) amplitude
        dlogSFR_amp = np.zeros((n_gal, n_col))

        if theta_sfh['sigma_corr'] > 0.: # if there's correlation between dlogMhalo and dlogSFR
            
            _dMhalos = np.zeros((n_gal, n_col)) # testing purposes

            # calculate dMhalo
            dlogSFR = np.zeros((n_gal, SHsnaps['nsnap0']-1)) # n_gal x (nsnap0 - 1) matrix

            for nsnap in range(1, SHsnaps['nsnap0'])[::-1]: 
                if nsnap == 1: 
                    mhalo_later = SHsnaps['halo.m'][indices]
                else: 
                    mhalo_later = SHsnaps['halo.m.snap'+str(nsnap)][indices]
                mhalo_early = SHsnaps['halo.m.snap'+str(nsnap+1)][indices]
                
                # log M_h(t_i) - log M_h(t_i+1) = log(M_h(t_i) / M_h(t_i+1))
                # (M_h(t_i+1)-M_h(t_i))/M_h(t_i+1) = 1. - M_h(t_i)/M_h(t_i+1)
                f_dMh = 1. - 10**(mhalo_early - mhalo_later)
                ibins = np.digitize(mhalo_later, 
                        np.arange(mhalo_later.min(), mhalo_later.max()+0.2, 0.2))
            
                #fig = plt.figure(1, figsize=(8*np.int(np.ceil(np.float(ibins.max())/3.)+1.), 15))
                for ibin in range(1, ibins.max()+1): 
                    inbin = np.where(ibins == ibin)
                    n_bin = len(inbin[0])
                     
                    isort = np.argsort(-1. * f_dMh[inbin])

                    irank = np.zeros(n_bin)
                    irank[isort] = (np.arange(n_bin) + 0.5)/np.float(n_bin)
                    
                    # i_rank = 1/2 (1 - erf(x/sqrt(2)))
                    # x = (SFR - avg_SFR)/sigma_SFR
                    # dSFR = sigma_SFR * sqrt(2) * erfinv(1 - 2 i_rank)
                    dlogSFR[inbin, SHsnaps['nsnap0'] - nsnap - 1] = \
                            theta_sfh['sigma_corr'] * 1.41421356 * erfinv(1. - 2. * irank)
                    
                    # test assembly bias 
                    #sub = fig.add_subplot(3, np.int(np.ceil(np.float(ibins.max())/3.)+1), ibin)
                    #sub.scatter(f_dMh[inbin], 0.3 * np.random.randn(n_bin), c='k')
                    #sub.scatter(f_dMh[inbin], dlogSFR[inbin, SHsnaps['nsnap0'] - nsnap - 1] + \
                    #                np.sqrt(0.3**2 - theta_sfh['sigma_corr']**2) * np.random.randn(n_bin), c='r', lw=0)
                    #sub.set_xlim([-1., 1.])
                    #sub.set_ylim([-1., 1.])
                # test assembly bias  implementation 
                #sub = fig.add_subplot(3, np.int(np.ceil(np.float(ibins.max())/3.)+1), ibin+1)
                #sub.scatter(f_dMh, 0.3 * np.random.randn(len(f_dMh)), c='k')
                #sub.scatter(f_dMh, dlogSFR[:, SHsnaps['nsnap0'] - nsnap - 1] + \
                #        np.sqrt(0.3**2 - theta_sfh['sigma_corr']**2) * \
                #        np.random.randn(len(f_dMh)), c='r', lw=0)
                #sub.set_xlim([-1., 1.])
                #sub.set_ylim([-1., 1.])
                ##print np.std(dlogSFR[inbin, SHsnaps['nsnap0'] - nsnap - 1])
                #fig.text(0.5, 0.04, '$\Delta M_h / M_h$', ha='center')
                #fig.text(0.04, 0.5, '$\Delta$ log SFR', va='center', rotation='vertical')
                #fig.savefig('abias_testing.png', bbox_inches='tight')
                #raise ValueError

            t_snaps = UT.t_nsnap(range(1, SHsnaps['nsnap0']+1)[::-1])

            for igal in range(n_gal): 
                i_tbins = np.digitize(tsteps[igal, range(n_col)], t_snaps) 
                i_tbins = i_tbins.clip(0, len(t_snaps)-1) # clip out of range indices
                
                #for n in range(n_col): 
                #    print t_snaps[i_tbins[n] - 1], '<= ', tsteps[igal, n], ' <',  t_snaps[i_tbins[n]]
                dlogSFR_amp[igal, range(n_col)] = dlogSFR[igal, i_tbins-1]
                
                #_dMhalos[igal, range(n_col)] = dMhalos[igal, i_tbins-1]

        # add in intrinsic scatter
        dlogSFR_corr = dlogSFR_amp
        dlogSFR_int = np.random.randn(n_gal, n_col) * np.sqrt(theta_sfh['sigma_tot']**2 - theta_sfh['sigma_corr']**2) 
        dlogSFR_amp += dlogSFR_int

        #dlogSFR_amp[:,0] = SHsnaps['sfr0'][indices] - mu_sfr0
        SHsnaps['sfr0'][indices] = mu_sfr0 + dlogSFR_amp[:,0]
        
        #for i_col in range(n_col): 
        #    plt.scatter(_dMhalos[range(10000), i_col], 0.3 * np.random.randn(10000), c='k')
        #    plt.scatter(_dMhalos[range(10000), i_col], dlogSFR_amp[range(10000), i_col], lw=0, c='r')
        #    plt.show()

        F_sfr = _logSFR_dSFR_tsteps
        
        sfr_kwargs = {'dlogSFR_amp': dlogSFR_amp, 'tsteps': tsteps,'theta_sfms': theta_sfms, 
                'dlogSFR_corr': dlogSFR_corr} #'dMhalos': _dMhalos, 
    
    elif theta_sfh['name'] == 'random_step_abias_dumb': 
        # random steps with assembly bias done in a dumb way 
        if 'dt_min' not in theta_sfh: 
            raise ValueError
        if 'dt_max' not in theta_sfh: 
            raise ValueError
                
        # Random step function duty cycle 
        del_t_max = UT.t_nsnap(1) - UT.t_nsnap(SHsnaps['nsnap0'])#SHsnaps['nsnap_start'][indices].max()) 
        
        # the range of the steps 
        tshift_min = theta_sfh['dt_min'] 
        tshift_max = theta_sfh['dt_max'] 

        # get the times when the amplitude changes 
        n_col = int(np.ceil(del_t_max/tshift_min))+1  # number of columns 
        n_gal = len(indices)

        tshift = np.zeros((n_gal, n_col))
        tshift[:,1:] = np.random.uniform(tshift_min, tshift_max, size=(n_gal, n_col-1))
        tsteps = np.cumsum(tshift , axis=1) + np.tile(UT.t_nsnap(SHsnaps['nsnap_start'][indices]), (n_col, 1)).T
        del tshift
        
        assert tsteps[range(n_gal), n_col-1].min() > UT.t_nsnap(1)
        
        # calculate d(logSFR) amplitude
        dlogSFR_amp = np.zeros((n_gal, n_col))

        if theta_sfh['sigma_corr'] > 0.: # if there's correlation between dlogMhalo and dlogSFR
            # calculate dlogSFR with assembly bias 
            # dlogSFR = 
            #           nsnap0-1, nsnap0-2, ..., 2, 1
            #   gal 1 [                      
            #   gal 2 [                      
            #   ...
            #gal ngal [                      

            dlogSFR = np.zeros((n_gal, SHsnaps['nsnap0']-1)) # n_gal x (nsnap0 - 1) matrix

            for nsnap in range(1, SHsnaps['nsnap0'])[::-1]: 
                if nsnap == 1: 
                    mhalo_later = SHsnaps['halo.m'][indices]
                else: 
                    mhalo_later = SHsnaps['halo.m.snap'+str(nsnap)][indices]
                mhalo_early = SHsnaps['halo.m.snap'+str(nsnap+1)][indices]
                
                # log M_h(t_i) - log M_h(t_i+1) = log(M_h(t_i) / M_h(t_i+1))
                # (M_h(t_i+1)-M_h(t_i))/M_h(t_i+1) = 1. - M_h(t_i)/M_h(t_i+1)
                f_dMh = 1. - 10**(mhalo_early - mhalo_later)

                mh_bins = np.arange(mhalo_later.min(), mhalo_later.max()+0.2, 0.2)


                for i_m in range(len(mh_bins)-1): 
                    inbin = np.where(
                            (mhalo_later >= mh_bins[i_m]) & 
                            (mhalo_later < mh_bins[i_m+1]))
                    n_bin = len(inbin[0])
                     
                    isort = np.argsort(-1. * f_dMh[inbin])

                    irank = np.zeros(n_bin)
                    irank[isort] = (np.arange(n_bin) + 0.5)/np.float(n_bin)
                    
                    # i_rank = 1/2 (1 - erf(x/sqrt(2)))
                    # x = (SFR - avg_SFR)/sigma_SFR
                    # dSFR = sigma_SFR * sqrt(2) * erfinv(1 - 2 i_rank)
                    dlogSFR[inbin, SHsnaps['nsnap0'] - nsnap - 1] = \
                            theta_sfh['sigma_corr'] * 1.41421356 * erfinv(1. - 2. * irank)
            
            # nsnap0, nsnap0-1, nsnap0-2, ... ,2, 1 
            t_snaps = UT.t_nsnap(range(1, SHsnaps['nsnap0']+1)[::-1])

            for igal in range(n_gal): 
                # each galaxy's timesteps 
                #i_tbins = np.digitize(tsteps[igal, range(n_col)], t_snaps) 
                #i_tbins = i_tbins.clip(0, len(t_snaps)-1) # clip out of range indices
                igal_tsteps = tsteps[igal, range(n_col)]
                i_tbins = np.digitize(igal_tsteps, t_snaps) 
            
                inrange = np.where((i_tbins < len(t_snaps)) & (i_tbins > 0)) 
                dlogSFR_amp_i = np.zeros(n_col)
                dlogSFR_amp_i[inrange] = dlogSFR[igal, i_tbins[inrange]-1]

                #dlogSFR_amp[igal, range(n_col)] = dlogSFR[igal, i_tbins-1]
                #_dMhalos[igal, range(n_col)] = dMhalos[igal, i_tbins-1]

        # add in intrinsic scatter
        #dlogSFR_corr = dlogSFR_amp
        dlogSFR_int = np.random.randn(n_gal, n_col) * np.sqrt(theta_sfh['sigma_tot']**2 - theta_sfh['sigma_corr']**2) 
        dlogSFR_amp += dlogSFR_int

        #dlogSFR_amp[:,0] = SHsnaps['sfr0'][indices] - mu_sfr0
        SHsnaps['sfr0'][indices] = mu_sfr0 + dlogSFR_amp[:,0]
        
        #for i_col in range(n_col): 
        #    plt.scatter(_dMhalos[range(10000), i_col], 0.3 * np.random.randn(10000), c='k')
        #    plt.scatter(_dMhalos[range(10000), i_col], dlogSFR_amp[range(10000), i_col], lw=0, c='r')
        #    plt.show()

        F_sfr = _logSFR_dSFR_tsteps
        
        sfr_kwargs = {'dlogSFR_amp': dlogSFR_amp, 'tsteps': tsteps,'theta_sfms': theta_sfms}
                #, 'dlogSFR_corr': dlogSFR_corr} #'dMhalos': _dMhalos, 
    
    elif theta_sfh['name'] == 'random_step_abias2': 
        # random steps with assembly bias with control over timescale of halo formation 
        if 'dt_min' not in theta_sfh: 
            raise ValueError
        if 'dt_max' not in theta_sfh: 
            raise ValueError
                
        # Random step function duty cycle 
        del_t_max = UT.t_nsnap(1) - UT.t_nsnap(SHsnaps['nsnap0'])#SHsnaps['nsnap_start'][indices].max()) 
        
        # the range of the steps 
        tshift_min = theta_sfh['dt_min'] 
        tshift_max = theta_sfh['dt_max'] 

        # get the times when the amplitude changes 
        n_col = int(np.ceil(del_t_max/tshift_min))+1  # number of columns 
        n_gal = len(indices)

        tshift = np.zeros((n_gal, n_col))
        tshift[:,1:] = np.random.uniform(tshift_min, tshift_max, size=(n_gal, n_col-1))
        tsteps = np.cumsum(tshift , axis=1) + np.tile(UT.t_nsnap(SHsnaps['nsnap_start'][indices]), (n_col, 1)).T
        del tshift
        
        assert tsteps[range(n_gal), n_col-1].min() > UT.t_nsnap(1)
        
        # calculate d(logSFR) amplitude
        dlogSFR_amp = np.zeros((n_gal, n_col))

        if theta_sfh['sigma_corr'] > 0.: # if there's correlation between dlogMhalo and dlogSFR
            # calculate dlogSFR with assembly bias 
            # dlogSFR = 
            #           nsnap0-1, nsnap0-2, ..., 2, 1
            #   gal 1 [                      
            #   gal 2 [                      
            #   ...
            #gal ngal [                      
            
            t_snaps = UT.t_nsnap(range(1, SHsnaps['nsnap0']+1)[::-1])

            t_abias = theta_sfh['t_abias'] # assembly bias timescale 
            n_abias = np.int(np.ceil((UT.t_nsnap(1) - UT.t_nsnap(SHsnaps['nsnap0']))/t_abias))
            t_step_abias = np.array([UT.t_nsnap(1) - (n_abias - i_abias) * t_abias 
                for i_abias in range(n_abias+1)])
            assert t_step_abias[0] < UT.t_nsnap(SHsnaps['nsnap0'])
            t_step_abias[0] = UT.t_nsnap(SHsnaps['nsnap0'])

            # M_halo of the galaxies in the snapshots 
            Mh_snaps = np.zeros((n_gal, SHsnaps['nsnap0'])) 
            for ii, isnap in enumerate(range(2, SHsnaps['nsnap0']+1)[::-1]): 
                Mh_snaps[:,ii] = SHsnaps['halo.m.snap'+str(isnap)][indices]
            Mh_snaps[:,-1] = SHsnaps['halo.m'][indices]

            Mh_abias = np.zeros((n_gal, len(t_step_abias)))
            for igal in range(n_gal): 
                Mh_snaps_i = 10**Mh_snaps[igal, range(SHsnaps['nsnap0'])]
                Mh_abias[igal, 0] = Mh_snaps_i[0]
                Mh_abias[igal, -1] = Mh_snaps_i[-1]
                Mh_abias[igal, 1:-1] = np.interp(t_step_abias[1:-1], t_snaps, Mh_snaps_i)
            Mh_abias = np.log10(Mh_abias)

            dlogSFR = np.zeros((Mh_abias.shape[0], Mh_abias.shape[1]-1))
            #dlogSFR = np.zeros((n_gal, SHsnaps['nsnap0']-1)) # n_gal x (nsnap0 - 1) matrix
            for ii in range(dlogSFR.shape[1]): 
                # log M_h(t_i) - log M_h(t_i+1) = log(M_h(t_i) / M_h(t_i+1))
                # (M_h(t_i+1)-M_h(t_i))/M_h(t_i+1) = 1. - M_h(t_i)/M_h(t_i+1)
                mhalo_later = Mh_abias[:, ii+1]
                mhalo_early = Mh_abias[:, ii]
                
                f_dMh = 1. - 10**(mhalo_early - mhalo_later)

                mh_bins = np.arange(mhalo_later.min(), mhalo_later.max()+0.2, 0.2)

                #fig = plt.figure(1, figsize=(8*np.int(np.ceil(np.float(len(mh_bins))/3.)+1.), 15))
                for i_m in range(len(mh_bins)-1): 
                    inbin = np.where((mhalo_later >= mh_bins[i_m]) & 
                            (mhalo_later < mh_bins[i_m+1]))
                    n_bin = len(inbin[0])
                     
                    isort = np.argsort(-1. * f_dMh[inbin])

                    irank = np.zeros(n_bin)
                    irank[isort] = (np.arange(n_bin) + 0.5)/np.float(n_bin)
                    
                    # i_rank = 1/2 (1 - erf(x/sqrt(2)))
                    # x = (SFR - avg_SFR)/sigma_SFR
                    # dSFR = sigma_SFR * sqrt(2) * erfinv(1 - 2 i_rank)
                    dlogSFR[inbin, ii] = \
                            theta_sfh['sigma_corr'] * 1.41421356 * erfinv(1. - 2. * irank)
                    
                    # test assembly bias 
                    #sub = fig.add_subplot(3, np.int(np.ceil(np.float(len(mh_bins))/3.)+1), i_m+1)
                    #sub.scatter(f_dMh[inbin], 0.3 * np.random.randn(n_bin), c='k')
                    #sub.scatter(f_dMh[inbin], dlogSFR[inbin, ii] + \
                    #                np.sqrt(0.3**2 - theta_sfh['sigma_corr']**2) * np.random.randn(n_bin), c='r', lw=0)
                    #sub.set_xlim([-1., 1.])
                    #sub.set_ylim([-1., 1.])
                # test assembly bias  implementation 
                #sub = fig.add_subplot(3, np.int(np.ceil(np.float(len(mh_bins))/3.)+1), i_m+2)
                #sub.scatter(f_dMh, 0.3 * np.random.randn(len(f_dMh)), c='k')
                #sub.scatter(f_dMh, dlogSFR[:, ii] + \
                #        np.sqrt(0.3**2 - theta_sfh['sigma_corr']**2) * \
                #        np.random.randn(len(f_dMh)), c='r', lw=0)
                #sub.set_xlim([-1., 1.])
                #sub.set_ylim([-1., 1.])
                ##print np.std(dlogSFR[inbin, SHsnaps['nsnap0'] - nsnap - 1])
                #fig.text(0.5, 0.04, '$\Delta M_h / M_h$', ha='center')
                #fig.text(0.04, 0.5, '$\Delta$ log SFR', va='center', rotation='vertical')
                #fig.savefig('abias_testing.png', bbox_inches='tight')
                #raise ValueError
            
            for igal in range(n_gal): 
                # each galaxy's timesteps 
                #i_tbins = np.digitize(tsteps[igal, range(n_col)], t_snaps) 
                #i_tbins = i_tbins.clip(0, len(t_snaps)-1) # clip out of range indices
                igal_tsteps = tsteps[igal, range(n_col)]
                i_tbins = np.digitize(igal_tsteps, t_step_abias) 
            
                inrange = np.where((i_tbins < len(t_step_abias)) & (i_tbins > 0)) 
                dlogSFR_amp_i = np.zeros(n_col)
                dlogSFR_amp_i[inrange] = dlogSFR[igal, i_tbins[inrange]-1]

                dlogSFR_amp[igal, range(n_col)] = dlogSFR_amp_i 

        # add in intrinsic scatter
        #dlogSFR_corr = dlogSFR_amp
        dlogSFR_int = np.random.randn(n_gal, n_col) * np.sqrt(theta_sfh['sigma_tot']**2 - theta_sfh['sigma_corr']**2) 
        dlogSFR_amp += dlogSFR_int

        #dlogSFR_amp[:,0] = SHsnaps['sfr0'][indices] - mu_sfr0
        SHsnaps['sfr0'][indices] = mu_sfr0 + dlogSFR_amp[:,0]
        
        #for i_col in range(n_col): 
        #    plt.scatter(_dMhalos[range(10000), i_col], 0.3 * np.random.randn(10000), c='k')
        #    plt.scatter(_dMhalos[range(10000), i_col], dlogSFR_amp[range(10000), i_col], lw=0, c='r')
        #    plt.show()

        F_sfr = _logSFR_dSFR_tsteps
        
        sfr_kwargs = {'dlogSFR_amp': dlogSFR_amp, 'tsteps': tsteps,'theta_sfms': theta_sfms}
                #, 'dlogSFR_corr': dlogSFR_corr} #'dMhalos': _dMhalos, 
    
    elif theta_sfh['name'] == 'random_step_abias_delay_dz': 
        # random steps with assembly bias where SFH is correlated with 
        # halo mass growth over dz_dMh, dt_delay earlier 
        if 'dt_min' not in theta_sfh: 
            raise ValueError
        if 'dt_max' not in theta_sfh: 
            raise ValueError
        if 'sigma_corr' not in theta_sfh:  
            raise ValueError
        else: 
            if theta_sfh['sigma_corr'] <= 0.: 
                raise ValueError("no assembly bias; dont use this SFH")

        if UT.z_nsnap(SHsnaps['nsnap0']) + theta_sfh['dz_dMh'] > UT.z_nsnap(SHsnaps['nsnap0'] + 10):
            print UT.z_nsnap(SHsnaps['nsnap0']) + theta_sfh['dz_dMh'], UT.z_nsnap(SHsnaps['nsnap0'] + 10)
            raise ValueError

        # first calculate the time at which the SFH changes (random step function duty cycle)
        del_t_max = UT.t_nsnap(1) - UT.t_nsnap(SHsnaps['nsnap0'])#SHsnaps['nsnap_start'][indices].max()) 
        
        tshift_min = theta_sfh['dt_min'] # the range of the steps 
        tshift_max = theta_sfh['dt_max'] 

        # get the times when the amplitude changes 
        n_col = int(np.ceil(del_t_max/tshift_min))+1  # number of columns 
        n_gal = len(indices)

        tshift = np.zeros((n_gal, n_col), dtype=np.float32)
        if tshift_min == tshift_max: 
            tshift[:,1:] = np.repeat(tshift_min, n_gal * (n_col-1)).reshape((n_gal, n_col-1))
        else: 
            tshift[:,1:] = np.random.uniform(tshift_min, tshift_max, size=(n_gal, n_col-1))
        tsteps = np.cumsum(tshift , axis=1) + np.tile(UT.t_nsnap(SHsnaps['nsnap_start'][indices]), (n_col, 1)).T
        #tsteps = np.clip(tsteps.astype(np.float32), 0., UT.t_nsnap(1))
        del tshift

        # M_h of the galaxies throughout the snapshots 
        # Mh_snaps = ngal x nsnap0+3 matrix 
        Mh_snaps = np.zeros((n_gal, SHsnaps['nsnap0']+9))#, dtype=np.float32)
        Mh_snaps[:,0] =  SHsnaps['halo.m'][indices]
        for isnap in range(2, SHsnaps['nsnap0']+10): 
            Mh_snaps[:,isnap-1] = SHsnaps['halo.m.snap'+str(isnap)][indices]
        
        z_snaps = UT.z_nsnap(range(1, SHsnaps['nsnap0']+10))
        t_snaps = UT.t_nsnap(range(1, SHsnaps['nsnap0']+10))

        # f_dMh growth rate of halo from t_delay - dt  to t_delay
        f_dMh = np.zeros(tsteps.shape, dtype=np.float32)
        Mh_steps = np.zeros(tsteps.shape, dtype=np.float32) 
        for i_g in range(n_gal): 
            insim = np.where(Mh_snaps[i_g,:] > 0.)[0] # some halos did not exist far back
            Mh_snaps_i = 10**(Mh_snaps[i_g,insim] - 10.) # scaled by 10^10 for ease
            t_snaps_i = t_snaps[insim]

            # SFH is correlated with halo growth dt_delay earlier  
            tstep_i = tsteps[i_g,:]
            tdelay_i = tstep_i - theta_sfh['dt_delay']
            
            # note that x has to be monotonically for np.interp 
            Mh_steps[i_g,:] = np.interp(tstep_i, t_snaps_i[::-1], Mh_snaps_i[::-1])
            Mh_delay  = np.interp(tdelay_i, t_snaps_i[::-1], Mh_snaps_i[::-1])
            Mh_delay_dz = np.interp(UT.z_of_t(tdelay_i) + theta_sfh['dz_dMh'], z_snaps[insim], Mh_snaps_i)

            f_dMh[i_g,:] = 1. - Mh_delay_dz / Mh_delay 
            #plt.plot(t_snaps_i[::-1], Mh_snaps_i[::-1], c='k', lw=2, ls='--')
            #plt.scatter(tdelay_i, Mh_delay, c='b', lw=0, s=40)
            #print 'tdelay = ', tdelay_i, UT.z_of_t(tdelay_i)
            #print 'zdelay+dz = ', UT.z_of_t(tdelay_i) + theta_sfh['dz_dMh']
            #print 't(zdelay+dz) = ', UT.t_of_z(UT.z_of_t(tdelay_i) + theta_sfh['dz_dMh'])

            #plt.scatter(UT.t_of_z(UT.z_of_t(tdelay_i) + theta_sfh['dz_dMh']), Mh_delay_dz, c='r', lw=0, s=40) 
            #plt.show()
            #if i_g > 10:
            #    raise ValueError


        # calculate the d(log SFR) amplitude at t_steps 
        dlogSFR_amp = np.zeros(f_dMh.shape, dtype=np.float32)
        for ii in range(f_dMh.shape[1]): 
            f_dMh_i = f_dMh[:,ii]
            Mh_step = np.log10(Mh_steps[:,ii]) + 10.
            mh_bins = np.arange(Mh_step.min(), Mh_step.max()+0.2, 0.2)
            
            #fig = plt.figure(1, figsize=(8*np.int(np.ceil(np.float(len(mh_bins))/3.)+1.), 15))
            for i_m in range(len(mh_bins)-1): 
                inbin = np.where((Mh_step >= mh_bins[i_m]) & 
                        (Mh_step < mh_bins[i_m+1]))
                n_bin = len(inbin[0])
                 
                isort = np.argsort(-1. * f_dMh_i[inbin])

                irank = np.zeros(n_bin)
                irank[isort] = (np.arange(n_bin) + 0.5)/np.float(n_bin)
                
                # i_rank = 1/2 (1 - erf(x/sqrt(2)))
                # x = (SFR - avg_SFR)/sigma_SFR
                # dSFR = sigma_SFR * sqrt(2) * erfinv(1 - 2 i_rank)
                dlogSFR_amp[inbin, ii] = \
                        theta_sfh['sigma_corr'] * 1.41421356 * erfinv(1. - 2. * irank) 

                #sub = fig.add_subplot(3, np.int(np.ceil(np.float(len(mh_bins))/3.)+1), i_m+1)
                #sub.scatter(f_dMh_i[inbin], 0.3 * np.random.randn(n_bin), c='k')
                #sub.scatter(f_dMh_i[inbin], dlogSFR_amp[inbin, ii] + \
                #        np.sqrt(0.3**2 - theta_sfh['sigma_corr']**2) * np.random.randn(n_bin), c='r', lw=0)
                #sub.set_xlim([-1., 1.])
                #sub.set_ylim([-1., 1.])

            #plt.show()
            #raise ValueError
        del f_dMh
        del Mh_steps
        
        # add in intrinsic scatter
        dlogSFR_int = np.random.randn(n_gal, n_col) * np.sqrt(theta_sfh['sigma_tot']**2 - theta_sfh['sigma_corr']**2) 
        dlogSFR_amp += dlogSFR_int

        #dlogSFR_amp[:,0] = SHsnaps['sfr0'][indices] - mu_sfr0
        SHsnaps['sfr0'][indices] = mu_sfr0 + dlogSFR_amp[:,0]
        
        F_sfr = _logSFR_dSFR_tsteps
        
        sfr_kwargs = {'dlogSFR_amp': dlogSFR_amp, 'tsteps': tsteps,'theta_sfms': theta_sfms}
    
    elif theta_sfh['name'] == 'random_step_abias_delay_dt': 
        # random steps with assembly bias where SFH is correlated with 
        # halo mass growth over dz_dMh, dt_delay earlier 
        if 'dt_min' not in theta_sfh: 
            raise ValueError
        if 'dt_max' not in theta_sfh: 
            raise ValueError
        if 'sigma_corr' not in theta_sfh:  
            raise ValueError
        else: 
            if theta_sfh['sigma_corr'] <= 0.: 
                raise ValueError("no assembly bias; dont use this SFH")

        if theta_sfh['dt_dMh'] == 0.: 
            raise ValueError("no assembly bias; dont use this SFH")


        if UT.t_nsnap(SHsnaps['nsnap0']) + theta_sfh['dt_dMh'] < UT.t_nsnap(SHsnaps['nsnap0'] + 10):
            print UT.t_nsnap(SHsnaps['nsnap0']) + theta_sfh['dt_dMh'], UT.t_nsnap(SHsnaps['nsnap0'] + 10)
            raise ValueError

        # first calculate the time at which the SFH changes (random step function duty cycle)
        del_t_max = UT.t_nsnap(1) - UT.t_nsnap(SHsnaps['nsnap0'])#SHsnaps['nsnap_start'][indices].max()) 
        
        tshift_min = theta_sfh['dt_min'] # the range of the steps 
        tshift_max = theta_sfh['dt_max'] 

        # get the times when the amplitude changes 
        n_col = int(np.ceil(del_t_max/tshift_min))+1  # number of columns 
        n_gal = len(indices)

        tshift = np.zeros((n_gal, n_col), dtype=np.float32)
        if tshift_min == tshift_max: 
            tshift[:,1:] = np.repeat(tshift_min, n_gal * (n_col-1)).reshape((n_gal, n_col-1))
        else: 
            tshift[:,1:] = np.random.uniform(tshift_min, tshift_max, size=(n_gal, n_col-1))
        tsteps = np.cumsum(tshift , axis=1) + np.tile(UT.t_nsnap(SHsnaps['nsnap_start'][indices]), (n_col, 1)).T
        #tsteps = np.clip(tsteps.astype(np.float32), 0., UT.t_nsnap(1))
        del tshift

        # M_h of the galaxies throughout the snapshots 
        # Mh_snaps = ngal x nsnap0+3 matrix 
        Mh_snaps = np.zeros((n_gal, SHsnaps['nsnap0']+9))#, dtype=np.float32)
        Mh_snaps[:,0] =  SHsnaps['halo.m'][indices]
        for isnap in range(2, SHsnaps['nsnap0']+10): 
            Mh_snaps[:,isnap-1] = SHsnaps['halo.m.snap'+str(isnap)][indices]
        
        z_snaps = UT.z_nsnap(range(1, SHsnaps['nsnap0']+10))
        t_snaps = UT.t_nsnap(range(1, SHsnaps['nsnap0']+10))

        # f_dMh growth rate of halo from t_delay - dt  to t_delay
        f_dMh = np.zeros(tsteps.shape, dtype=np.float32)
        Mh_steps = np.zeros(tsteps.shape, dtype=np.float32) 
        for i_g in range(n_gal): 
            insim = np.where(Mh_snaps[i_g,:] > 0.)[0] # some halos did not exist far back
            Mh_snaps_i = 10**(Mh_snaps[i_g,insim] - 10.) # scaled by 10^10 for ease
            t_snaps_i = t_snaps[insim]

            # SFH is correlated with halo growth dt_delay earlier  
            tstep_i = tsteps[i_g,:]
            tdelay_i = tstep_i - theta_sfh['dt_delay']
            
            # note that x has to be monotonically increasing for np.interp 
            Mh_steps[i_g,:] = np.interp(tstep_i, t_snaps_i[::-1], Mh_snaps_i[::-1])
            Mh_delay  = np.interp(tdelay_i, t_snaps_i[::-1], Mh_snaps_i[::-1])
            Mh_delay_dt = np.interp(tdelay_i - theta_sfh['dt_dMh'], t_snaps_i[::-1], Mh_snaps_i[::-1])

            f_dMh[i_g,:] = 1. - Mh_delay_dt / Mh_delay 

            #print 'tdelay = ', tdelay_i, UT.z_of_t(tdelay_i)
            #print 'zdelay+dz = ', UT.z_of_t(tdelay_i) + theta_sfh['dz_dMh']
            #print 't(zdelay+dz) = ', UT.t_of_z(UT.z_of_t(tdelay_i) + theta_sfh['dz_dMh'])
            #for tt in tstep_i:  
            #    plt.vlines(tt, np.log10(Mh_delay_dt.min())+10, 
            #            np.log10(Mh_steps[i_g,:].max())+10, linestyle='--')
            #plt.plot(t_snaps_i[::-1], np.log10(Mh_snaps_i[::-1])+10, c='k', lw=2, ls='--', 
            #        label='$M_h$ snapshots')
            #plt.scatter(tdelay_i, np.log10(Mh_delay)+10, c='b', lw=0, s=40, label='$M_h(t_{delay})$')
            #plt.scatter(tdelay_i - theta_sfh['dt_dMh'], np.log10(Mh_delay_dt)+10, c='r', lw=0, s=40, label='$M_h(t_{delay}+\Delta t)$')
            #plt.show()
            #if i_g > 20:
            #    raise ValueError

        # calculate the d(log SFR) amplitude at t_steps 
        dlogSFR_amp = np.zeros(f_dMh.shape, dtype=np.float32)
        for ii in range(f_dMh.shape[1]): 
            f_dMh_i = f_dMh[:,ii]
            Mh_step = np.log10(Mh_steps[:,ii]) + 10.
            mh_bins = np.arange(Mh_step.min(), Mh_step.max()+0.2, 0.2)
            
            #fig = plt.figure(1, figsize=(8*np.int(np.ceil(np.float(len(mh_bins))/3.)+1.), 15))
            for i_m in range(len(mh_bins)-1): 
                inbin = np.where((Mh_step >= mh_bins[i_m]) & 
                        (Mh_step < mh_bins[i_m+1]))
                n_bin = len(inbin[0])
                 
                isort = np.argsort(-1. * f_dMh_i[inbin])

                irank = np.zeros(n_bin)
                irank[isort] = (np.arange(n_bin) + 0.5)/np.float(n_bin)
                
                # i_rank = 1/2 (1 - erf(x/sqrt(2)))
                # x = (SFR - avg_SFR)/sigma_SFR
                # dSFR = sigma_SFR * sqrt(2) * erfinv(1 - 2 i_rank)
                dlogSFR_amp[inbin, ii] = \
                        theta_sfh['sigma_corr'] * 1.41421356 * erfinv(1. - 2. * irank) 

                #sub = fig.add_subplot(3, np.int(np.ceil(np.float(len(mh_bins))/3.)+1), i_m+1)
                #sub.scatter(f_dMh_i[inbin], 0.3 * np.random.randn(n_bin), c='k')
                #sub.scatter(f_dMh_i[inbin], dlogSFR_amp[inbin, ii] + \
                #        np.sqrt(0.3**2 - theta_sfh['sigma_corr']**2) * np.random.randn(n_bin), c='r', lw=0)
                #sub.set_xlim([-1., 1.])
                #sub.set_ylim([-1., 1.])

            #plt.show()
            #raise ValueError
        del f_dMh
        del Mh_steps
        
        # add in intrinsic scatter
        dlogSFR_int = np.random.randn(n_gal, n_col) * np.sqrt(theta_sfh['sigma_tot']**2 - theta_sfh['sigma_corr']**2) 
        dlogSFR_amp += dlogSFR_int

        #dlogSFR_amp[:,0] = SHsnaps['sfr0'][indices] - mu_sfr0
        SHsnaps['sfr0'][indices] = mu_sfr0 + dlogSFR_amp[:,0]
        
        F_sfr = _logSFR_dSFR_tsteps
        
        sfr_kwargs = {'dlogSFR_amp': dlogSFR_amp, 'tsteps': tsteps,'theta_sfms': theta_sfms}
    
    else:
        raise NotImplementedError

    return F_sfr, sfr_kwargs


def _logSFR_dSFR(logmm, zz, dSFR=None, theta_sfms=None): 
    return SFR_sfms(logmm, zz, theta_sfms) + dSFR


def _logSFR_dSFR_tsteps(logmm, zz, tsteps=None, dlogSFR_amp=None, theta_sfms=None, **other_kwargs): 
    ''' Rough test verify that this does indeed produce a step function that 
    changes at t_steps 
    '''
    # log(SFR) of SF MS 
    logsfr_sfms = SFR_sfms(logmm, zz, theta_sfms)

    # dlog(SFR) 
    tt = UT.t_of_z(zz, deg=6) # t_cosmic(zz)
    
    # get the amplitude of the 
    ishift = np.abs(tsteps - tt).argmin(axis=1)
    closest = tsteps[range(len(ishift)),ishift]
    after = np.where(closest > tt)
    ishift[after] -= 1
    dlogsfr = dlogSFR_amp[range(len(ishift)),ishift]
    ## testing 
    #gals = np.random.choice(range(len(logmm)), 10) 
    #for igal in gals: 
    #    plt.plot(tsteps[igal,:], dlogSFR_amp[igal,:], c='k') 
    #    plt.scatter(np.repeat(tt, 2), np.repeat(dlogsfr[igal], 2), c='r', lw=0)
    #    plt.show() 
    return logsfr_sfms + dlogsfr


def ODE_Euler(dydt, init_cond, t_arr, delt, **func_args): 
    '''
    NOTE t_arr[0] = t0!!! 
    '''
    # t where we will evaluate 
    t_eval = np.arange(t_arr.min(), t_arr.max()+delt, delt) 
    t_eval[-1] = t_arr[-1]

    indices = []
    for tt in t_arr[1:-1]:
        idx = np.argmin(np.abs(t_eval - tt))
        assert np.abs(t_eval[idx] - tt) < delt
        t_eval[idx] = tt
        indices.append(idx)
    indices.append(len(t_eval) - 1) 

    dts = t_eval[1:] - t_eval[:-1]

    y = init_cond.copy()
    y_out = [init_cond.copy()]

    for it in range(len(dts)):
        dy = dts[it] * dydt(y, t_eval[it], **func_args)
        y += dy 

        if it+1 in indices: 
            y_out.append(y.copy())

    return np.array(y_out)


def ODE_RK4(dydt, init_cond, t_arr, delt, **func_args): 
    '''
    '''
    # t where we will evaluate 
    t_eval = np.arange(t_arr.min(), t_arr.max()+delt, delt) 
    t_eval[-1] = t_arr[-1]

    indices = [0]
    for tt in t_arr[1:-1]:
        idx = np.argmin(np.abs(t_eval - tt))
        t_eval[idx] = tt
        indices.append(idx)
    indices.append(len(t_eval) - 1) 

    dts = t_eval[1:] - t_eval[:-1]

    y = init_cond.copy()
    y_out = [init_cond.copy()]

    for it in range(len(dts)):
        k1 = dts[it] * dydt(y, t_eval[it], **func_args)
        k2 = dts[it] * dydt(y + 0.5 * k1, t_eval[it] + 0.5 * dts[it], **func_args)
        k3 = dts[it] * dydt(y + 0.5 * k2, t_eval[it] + 0.5 * dts[it], **func_args)
        k4 = dts[it] * dydt(y + k3, t_eval[it] + dts[it], **func_args)
        y += (k1 + 2.*k2 + 2.*k3 + k4)/6.

        if it+1 in indices: 
            y_out.append(y.copy())
    
    return np.array(y_out)


def dlogMdt(logMstar, t, logsfr_M_z=None, f_retain=None, zoft=None, **sfr_kwargs): 
    ''' Integrand d(logM)/dt for solving the ODE 

    d(logM)/dz = SFR(logM, z) * dt/dz * 10^9 / (M ln(10))

    Parameters
    ----------
    logsfr_M_t : (function) 
        log( SFR(logM, z) )

    logMstar : (array)
        Array of stellar masses

    t : (float) 
        Cosmic time 

    '''
    tmp = logsfr_M_z(logMstar, zoft(t), **sfr_kwargs) + 8.6377843113005373 - logMstar
    dlogMdz = f_retain * np.power(10, tmp) 

    return dlogMdz 


def dlogMdt_scipy(logMstar, t, logsfr_M_z, f_retain, zoft, sfr_kwargs): 
    ''' Integrand d(logM)/dt for solving the ODE 

    d(logM)/dz = SFR(logM, z) * dt/dz * 10^9 / (M ln(10))

    Parameters
    ----------
    logsfr_M_t : (function) 
        log( SFR(logM, z) )

    logMstar : (array)
        Array of stellar masses

    t : (float) 
        Cosmic time 

    '''
    tmp = logsfr_M_z(logMstar, zoft(t), **sfr_kwargs) + 8.6377843113005373 - logMstar
    dlogMdz = f_retain * np.power(10, tmp) 

    return dlogMdz 


def SFR_sfms(logm, z, theta): 
    ''' Average SFR of the SFMS as a function of logm at redshift z, i.e. log SFR(M*, z).
    The model takes the functional form of 
        log(SFR) = A * (log M* - logM_fid) + B * (z - z_fid) + C
    '''
    assert theta is not None 
    
    if 'name' not in theta.keys(): 
        raise ValueError
    
    if theta['name'] == 'flex': # this is a very flexible SFMS
        if 'mslope' not in theta.keys(): 
            raise ValueError
        if 'zslope' not in theta.keys(): 
            raise ValueError
        return theta['mslope'] * (logm - 10.5) + theta['zslope'] * (z - 0.05) - 0.11 
    elif theta['name'] == 'anchored':
        if 'amp' not in theta.keys(): 
            raise ValueError
        if 'slope' not in theta.keys(): 
            raise ValueError

        # in this prescription, log SFR_MS is anchored at z = 0 from SDSS DR7 central galaxy SFMS
        return (0.5757 * (logm - 10.5) - 0.13868) + \
                (z - 0.05) * (theta['slope'] * (logm - 10.5) + theta['amp']) 


def SSFR_sfms(logm, z, theta): 
    return SFR_sfms(logm, z, theta) - logm 
