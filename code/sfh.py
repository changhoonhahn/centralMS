'''



'''
import time
import numpy as np 

# --- local --- 
import util as UT
import observables as Obvs


def logSFR_initiate(SHsnaps, indices, theta_sfh=None, theta_sfms=None):
    '''
    '''
    if theta_sfh['name'] == 'constant_offset':
        # constant d_logSFR 
        mu_sfr0 = Obvs.SSFR_SFMS(SHsnaps['m.star0'][indices], 
                UT.z_nsnap(SHsnaps['nsnap_start'][indices]), theta_SFMS=theta_sfms) + SHsnaps['m.star0'][indices]
    
        F_sfr = _logSFR_dSFR 
        sfr_kwargs = {'dSFR': SHsnaps['sfr0'][indices] - mu_sfr0,  # offset
                'theta_sfms': theta_sfms}

    elif theta_sfh['name'] == 'corr_constant_offset': 
        # constant d_logSFR (assigned based on halo accretion rate) 
        mu_sfr0 = Obvs.SSFR_SFMS(SHsnaps['m.star0'][indices], 
                UT.z_nsnap(SHsnaps['nsnap_start'][indices]), theta_SFMS=theta_sfms) + SHsnaps['m.star0'][indices]

        #dSFR0 = SHsnaps['sfr0'][indices] - mu_sfr

        # now rank order it based on halo accretion rate
        dMhalo = SHsnaps['halo.m'][indices] - SHsnaps['halo.m0'][indices]
        
        m_kind = SHsnaps[theta_sfh['m.kind']+'0'][indices] # how do we bin the SFRs?
        dm_kind = theta_sfh['dm.kind'] # bins to rank order
    
        # scatter from noise -- subtract intrinsic assembly bias scatter from sig_SFMS 
        sig_noise = np.sqrt(Obvs.sigSSFR_SFMS(SHsnaps['m.star0'][indices])**2 - theta_sfh['sig_abias']**2)

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
        mu_sfr0 = Obvs.SSFR_SFMS(SHsnaps['m.star0'][indices], 
                UT.z_nsnap(SHsnaps['nsnap_start'][indices]), theta_SFMS=theta_sfms) + SHsnaps['m.star0'][indices]
                
        # Random step function duty cycle 
        del_t_max = UT.t_nsnap(1) - UT.t_nsnap(SHsnaps['nsnap_start'][indices].max()) 
        
        # the range of the steps 
        tshift_min = theta_sfh['dt_min'] 
        tshift_max = theta_sfh['dt_max'] 

        # get the times when the amplitude changes 
        n_col = int(np.ceil(del_t_max/tshift_min))  # number of columns 
        n_gal = len(indices)

        tshift = np.zeros((n_gal, n_col))
        tshift[:,1:] = np.random.uniform(tshift_min, tshift_max, size=(n_gal, n_col-1))
        tsteps = np.cumsum(tshift , axis=1) + np.tile(UT.t_nsnap(SHsnaps['nsnap_start'][indices]), (n_col, 1)).T
        del tshift

        dlogSFR_amp = np.random.randn(n_gal, n_col) * theta_sfh['sigma']
        dlogSFR_amp[:,0] = SHsnaps['sfr0'][indices] - mu_sfr0

        F_sfr = _logSFR_dSFR_tsteps
        
        sfr_kwargs = {'dlogSFR_amp': dlogSFR_amp, 'tsteps': tsteps,'theta_sfms': theta_sfms}
    
    elif theta_sfh['name'] == 'random_step_abias': 
        # random steps with assembly bias  
        if 'dt_min' not in theta_sfh: 
            raise ValueError
        if 'dt_max' not in theta_sfh: 
            raise ValueError
        mu_sfr0 = Obvs.SSFR_SFMS(SHsnaps['m.star0'][indices], 
                UT.z_nsnap(SHsnaps['nsnap_start'][indices]), theta_SFMS=theta_sfms) + SHsnaps['m.star0'][indices]
                
        # Random step function duty cycle 
        del_t_max = UT.t_nsnap(1) - UT.t_nsnap(SHsnaps['nsnap_start'][indices].max()) 
        
        # the range of the steps 
        tshift_min = theta_sfh['dt_min'] 
        tshift_max = theta_sfh['dt_max'] 

        # get the times when the amplitude changes 
        n_col = int(np.ceil(del_t_max/tshift_min))  # number of columns 
        n_gal = len(indices)

        tshift = np.zeros((n_gal, n_col))
        tshift[:,1:] = np.random.uniform(tshift_min, tshift_max, size=(n_gal, n_col-1))
        tsteps = np.cumsum(tshift , axis=1) + np.tile(UT.t_nsnap(SHsnaps['nsnap_start'][indices]), (n_col, 1)).T
        del tshift

        dlogSFR_amp = np.random.randn(n_gal, n_col) * theta_sfh['sigma']
        dlogSFR_amp[:,0] = SHsnaps['sfr0'][indices] - mu_sfr0

        F_sfr = _logSFR_dSFR_tsteps
        
        sfr_kwargs = {'dlogSFR_amp': dlogSFR_amp, 'tsteps': tsteps,'theta_sfms': theta_sfms}
    else:
        raise NotImplementedError

    return F_sfr, sfr_kwargs


def _logSFR_dSFR(logmm, zz, dSFR=None, theta_sfms=None): 
    return Obvs.SSFR_SFMS(logmm, zz, theta_SFMS=theta_sfms) + logmm + dSFR


def _logSFR_dSFR_tsteps(logmm, zz, tsteps=None, dlogSFR_amp=None, theta_sfms=None): 
    ''' 
    '''
    # log(SFR) of SF MS 
    logsfr_sfms = Obvs.SSFR_SFMS(logmm, zz, theta_SFMS=theta_sfms) + logmm

    # dlog(SFR) 
    tt = UT.t_of_z(zz, deg=6) # t_cosmic(zz)
    
    # get the amplitude of the 
    ishift = np.abs(tsteps - tt).argmin(axis=1)
    closest = tsteps[range(len(ishift)),ishift]
    after = np.where(closest > tt)
    ishift[after] -= 1
    dlogsfr = dlogSFR_amp[range(len(ishift)),ishift]

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