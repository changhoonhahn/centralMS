import time
import numpy as np 
from scipy.special import erfinv
from scipy.interpolate import interp1d

# --- local --- 
import util as UT
import corner as DFM 
import matplotlib.pyplot as plt


def logSFR_initiate(SHsnaps, indices, theta_sfh=None, theta_sfms=None, testing=False):
    ''' initiate log SFR function for Evolver.Evolve() method
    '''
    nsnap0 = SHsnaps['metadata']['nsnap0']
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
        del_t_max = UT.t_nsnap(1) - UT.t_nsnap(nsnap0) #'nsnap_start'][indices].max()) 
        
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
        # completely random amplitude that is sampled from a Gaussian with sig_logSFR
        # EXCEPT each adjacent timesteps have alternating sign amplitudes
        # time steps have width specified tduty 
        dt_tot= UT.t_nsnap(1) - UT.t_nsnap(nsnap0) # total cosmic time of evolution 

        tduty = theta_sfh['tduty'] # dutycycle time

        # get the times when the amplitude changes 
        n_col = int(np.ceil(dt_tot / tduty))+2  # number of columns 
        n_gal = len(indices)    # number of galaxies
        tshift = np.tile(tduty, (n_gal, n_col))
        tshift[:,0] = 0.
        tshift[:,1] = np.random.uniform(0., tduty, n_gal) 
        tsteps = np.cumsum(tshift , axis=1) + np.tile(UT.t_nsnap(SHsnaps['nsnap_start'][indices]), (n_col, 1)).T
        del tshift
        # make sure everything evolves properly until the end
        assert tsteps[range(n_gal), n_col-1].min() > UT.t_nsnap(1)
        
        # dlogSFR absolute amplitue 
        dlogSFR_amp = np.abs(np.random.randn(n_gal, n_col)) * theta_sfh['sigma']
        dlogSFR_amp[:,0] = SHsnaps['sfr0'][indices] - mu_sfr0 # dlogSFR at nsnap_start 

        # now make every other time step has fluctuating signs!
        pos = dlogSFR_amp[:,0] >= 0.
        neg = ~pos

        fluct = np.ones(n_col-1)
        fluct[2 * np.arange(int(np.ceil(float(n_col-1)/2.)))] *= -1.
        
        dlogSFR_amp[pos,1:] *= fluct 
        dlogSFR_amp[neg,1:] *= -1. * fluct

        # testing the distribution of delta SFR
        if testing: 
            for i in range(n_col):
                print('std of dlogSFR amp = %f' % np.std(dlogSFR_amp))
                plt.hist(dlogSFR_amp[:,i], range=(-1., 1.), 
                        linewidth=2, histtype='step', density=True, label='Step '+str(i)) 
                plt.legend()
                plt.xlabel(r'$\Delta\log\,\mathrm{SFR}$', fontsize=20)
                plt.xlim([-1., 1.])
            plt.savefig(''.join([UT.fig_dir(), 'random_step_fluct_test.png']), bbox_inches='tight')

        F_sfr = _logSFR_dSFR_tsteps
        sfr_kwargs = {'dlogSFR_amp': dlogSFR_amp, 'tsteps': tsteps,'theta_sfms': theta_sfms}

    elif theta_sfh['name'] == 'random_step_abias_dt': 
        # SFH where the amplitude w.r.t. the SFS is correlated with halo mass growth over 
        # tdyn(t). The amplitude is sampled at every time step specified by `tduty`
        dt_tot= UT.t_nsnap(1) - UT.t_nsnap(nsnap0) # total cosmic time of evolution 
        tduty = theta_sfh['tduty'] # dutycycle time
        sigma_int = np.sqrt(theta_sfh['sigma_tot']**2 - theta_sfh['sigma_corr']**2) # intrinsic sigma 

        # get the times when the amplitude changes 
        n_col = int(np.ceil(dt_tot / tduty))+2  # number of columns 
        n_gal = len(indices)    # number of galaxies
        tshift = np.tile(tduty, (n_gal, n_col))
        tshift[:,0] = 0.
        tshift[:,1] = np.random.uniform(0., tduty, n_gal) 
        tsteps = np.cumsum(tshift , axis=1) + \
                np.tile(UT.t_nsnap(SHsnaps['nsnap_start'][indices]), (n_col, 1)).T
        del tshift

        # M_h of the galaxies throughout the snapshots 
        Mh_snaps = np.zeros((n_gal, nsnap0+9), dtype=np.float32)
        Mh_snaps[:,0] = SHsnaps['halo.m'][indices]
        Mm_snaps = np.zeros((n_gal, nsnap0+9), dtype=np.float32)
        Mm_snaps[:,0] = SHsnaps['m.max'][indices]
        for isnap in range(2, nsnap0+10): 
            Mh_snaps[:,isnap-1] = SHsnaps['halo.m.snap'+str(isnap)][indices]
            Mm_snaps[:,isnap-1] = SHsnaps['m.max.snap'+str(isnap)][indices]
        
        t_snaps = UT.t_nsnap(range(1, nsnap0+10))

        # now we need to use these M_h to calculate the halo growth rates
        # at every time step t_i over the range t_i - dt_dMh to t_i. This means
        # we need to get M_h at both t_i and t_i-dt_dMh...
        f_dMh = np.zeros(tsteps.shape, dtype=np.float32) # growth rate of halos over t_i and t_i - dt_Mh
        Mh_ts = np.zeros(tsteps.shape, dtype=np.float32)
        Mm_ts = np.zeros(tsteps.shape, dtype=np.float32)
        iii, iiii = 0, 0 
        for i_g in range(n_gal): 
            in_sim = (Mm_snaps[i_g,:] > 0.)
            t_snaps_i = t_snaps[in_sim]
            Mh_snaps_i = np.power(10., Mh_snaps[i_g, in_sim]) 
            Mm_snaps_i = np.power(10., Mm_snaps[i_g, in_sim]) 
            
            t_i = tsteps[i_g,:] # t_i
            #t_imdt = tsteps[i_g,:] - theta_sfh['dt_dMh'] # t_i - dt_dMh
            t_imdt = t_i - UT.tdyn_t(t_i) # t_i - tdyn 

            Mh_ti = np.interp(t_i, t_snaps_i[::-1], Mh_snaps_i[::-1]) 
            Mh_timdt = np.interp(t_imdt, t_snaps_i[::-1], Mh_snaps_i[::-1]) 
            
            f_dMh[i_g,:] = Mh_ti / Mh_timdt #1. - Mh_timdt /  
            Mh_ts[i_g,:] = np.log10(Mh_ti)
            Mm_ti = np.interp(t_i, t_snaps_i[::-1], Mm_snaps_i[::-1]) 
            Mm_ts[i_g,:] = np.log10(Mm_ti)
            
            if testing: 
                if (SHsnaps['nsnap_start'][indices][i_g] == 15) and (iii < 10): 
                    fig = plt.figure(figsize=(8,4)) 
                    sub = fig.add_subplot(121)
                    sub.plot(t_snaps_i, Mh_snaps_i/Mh_snaps_i[0], c='k', ls='--') 
                    sub.fill_between([t_imdt[0], t_i[0]], [0., 0.], [1.2, 1.2], color='C0')
                    sub.fill_between([t_imdt[-2], t_i[-2]], [0., 0.], [1.2, 1.2], color='C1')
                    sub.vlines(UT.t_nsnap(15), 0., 1.2) 
                    sub.vlines(UT.t_nsnap(1),  0., 1.2) 
                    sub.set_xlim([0., 14.]) 
                    sub.set_ylim([0., 1.2]) 

                    sub = fig.add_subplot(122)
                    sub.plot(t_i, f_dMh[i_g,:], c='k', ls='--') 
                    sub.scatter([t_i[0]], [f_dMh[i_g,0]], c='C0')
                    sub.scatter([t_i[-2]], [f_dMh[i_g,-2]], c='C1')
                    sub.vlines(UT.t_nsnap(15), -0.2, 5.) 
                    sub.vlines(UT.t_nsnap(1), -0.2, 5.) 
                    sub.set_xlim([0., 14.]) 
                    sub.set_ylim([0., 5.]) 

                    fig.savefig(''.join([UT.fig_dir(), 'abiastest', str(iii), '.png']), bbox_inches='tight')
                    plt.close()
                    iii += 1 
                
                if in_sim[15]: 
                    if Mh_snaps_i[0] < Mh_snaps_i[14]: 
                        if iiii > 10: continue 
                        fig = plt.figure(figsize=(8,4)) 
                        sub = fig.add_subplot(121)
                        sub.plot(t_snaps_i, Mh_snaps_i/Mh_snaps_i[0], c='k', ls='--') 
                        sub.fill_between([t_imdt[0], t_i[0]], [0., 0.], [1.2, 1.2], color='C0')
                        sub.fill_between([t_imdt[-2], t_i[-2]], [0., 0.], [1.2, 1.2], color='C1')
                        sub.vlines(UT.t_nsnap(15), 0., 1.2) 
                        sub.vlines(UT.t_nsnap(1),  0., 1.2) 
                        sub.set_xlim([0., 14.]) 
                        sub.set_ylim([0., 1.2]) 

                        sub = fig.add_subplot(122)
                        sub.plot(t_i, f_dMh[i_g,:], c='k', ls='--') 
                        sub.scatter([t_i[0]], [f_dMh[i_g,0]], c='C0')
                        sub.scatter([t_i[-2]], [f_dMh[i_g,-2]], c='C1')
                        sub.vlines(UT.t_nsnap(15), -0.2, 1.) 
                        sub.vlines(UT.t_nsnap(1), -0.2, 1.) 
                        sub.set_xlim([0., 14.]) 
                        sub.set_ylim([0., 5.]) 
                        sub.text(0.05, 0.05, r'$\log\,M_h='+str(round(np.log10(Mh_snaps_i[0]),2))+'$',
                                fontsize=15, ha='left', va='bottom', transform=sub.transAxes)
                        fig.savefig(''.join([UT.fig_dir(), 'weird_abiastest', str(iiii), '.png']), bbox_inches='tight')
                        plt.close()
                        iiii += 1

        # calculate the d(log SFR) amplitude at t_steps 
        # at each t_step, for a given halo mass bin of 0.2 dex, 
        # rank order the SFRs and the halo growth rate 
        dlogSFR_amp = np.zeros(f_dMh.shape, dtype=np.float32)
        dlogMh = 0.1
        for ii in range(f_dMh.shape[1]): 
            f_dMh_i = f_dMh[:,ii]
            Mh_ti = Mm_ts[:,ii] 
            mh_bins = np.arange(Mh_ti.min(), Mh_ti.max(), dlogMh)
            #mh_bins = np.linspace(Mh_ti.min(), Mh_ti.max(), 100)
            
            if testing: 
                fig = plt.figure(1, figsize=(4*np.int(np.ceil(float(len(mh_bins))/3.)+1.), 8))
            for i_m in range(len(mh_bins)-1): 
                inbin = ((Mh_ti >= mh_bins[i_m]) & (Mh_ti < mh_bins[i_m]+dlogMh))
                #inbin = ((Mh_ti >= mh_bins[i_m]) & (Mh_ti < mh_bins[i_m+1]))
                n_bin = np.sum(inbin)
                 
                isort = np.argsort(-1. * f_dMh_i[inbin])

                irank = np.zeros(n_bin)
                irank[isort] = (np.arange(n_bin) + 0.5)/np.float(n_bin)
                
                # i_rank = 1/2 (1 - erf(x/sqrt(2)))
                # x = (SFR - avg_SFR)/sigma_SFR
                # dSFR = sigma_SFR * sqrt(2) * erfinv(1 - 2 i_rank)
                dlogSFR_amp[inbin, ii] = \
                        theta_sfh['sigma_corr'] * 1.41421356 * erfinv(1. - 2. * irank) 
                
                if testing: 
                    sub = fig.add_subplot(3, np.int(np.ceil(np.float(len(mh_bins))/3.)+1), i_m+1)
                    sub.scatter(f_dMh_i[inbin], 0.3 * np.random.randn(n_bin), c='k', s=2)
                    sub.scatter(f_dMh_i[inbin], dlogSFR_amp[inbin, ii] + sigma_int * np.random.randn(n_bin), c='r', s=2, lw=0)
                    sub.set_xlim([-0.5, 1.])
                    sub.set_ylim([-1., 1.])
            if testing: 
                plt.savefig(''.join([UT.fig_dir(), 'random_step_abias_dt_test', str(ii), '.png']), bbox_inches='tight')
                plt.close()

                fig = plt.figure(figsize=(8,4)) 
                sub = fig.add_subplot(121)
                DFM.hist2d(Mh_ti, f_dMh_i, levels=[0.68, 0.95], range=[[10., 14.],[0., 10.]], color='k', 
                        plot_datapoints=False, fill_contours=False, plot_density=True, ax=sub)
                sub.set_xlim([10., 14.]) 
                sub = fig.add_subplot(122)
                DFM.hist2d(Mh_ti, dlogSFR_amp[:,ii], levels=[0.68, 0.95], range=[[10., 14.],[-1., 1.]], color='k', 
                        plot_datapoints=False, fill_contours=False, plot_density=True, ax=sub)
                sub.set_xlim([10., 14.]) 
                fig.savefig(''.join([UT.fig_dir(), 'abias_fdMh', str(ii), '.png']), bbox_inches='tight')
                plt.close()

        del f_dMh
        del Mh_ts
        
        # add in intrinsic scatter
        dlogSFR_int = np.random.randn(n_gal, n_col) * sigma_int 
        dlogSFR_amp += dlogSFR_int
        SHsnaps['sfr0'][indices] = mu_sfr0 + dlogSFR_amp[:,0] # change SFR_0 so that it's assembly biased as well
        
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
