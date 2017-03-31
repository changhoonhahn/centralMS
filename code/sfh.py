'''



'''
import time
import numpy as np 

# --- local --- 
import util as UT
import observables as Obvs


def logSFR_wrapper(SHsnaps, indices, theta_sfh=None, theta_sfms=None):
    '''
    '''
    if theta_sfh['name'] == 'constant_offset':
        # constant d_logSFR 
        mu_sfr = Obvs.SSFR_SFMS(SHsnaps['m.star0'][indices], 
                UT.z_nsnap(SHsnaps['nsnap_start'][indices]), theta_SFMS=theta_sfms) + SHsnaps['m.star0'][indices]
    
        F_sfr = _logSFR_dSFR 
        sfr_kwargs = {'dSFR': SHsnaps['sfr0'][indices] - mu_sfr,  # offset
                'theta_sfms': theta_sfms}

    elif theta_sfh['name'] == 'corr_constant_offset': 
        # constant d_logSFR (assigned based on halo accretion rate) 
        mu_sfr = Obvs.SSFR_SFMS(SHsnaps['m.star0'][indices], 
                UT.z_nsnap(SHsnaps['nsnap_start'][indices]), theta_SFMS=theta_sfms) + SHsnaps['m.star0'][indices]

        #dSFR0 = SHsnaps['sfr0'][indices] - mu_sfr

        # now rank order it based on halo accretion rate
        dMhalo = SHsnaps['halo.m'][indices] - SHsnaps['halo.m0'][indices]
        
        m_kind = SHsnaps[theta_sfh['m.kind']+'0'][indices] # how do we bin the SFRs?
        dm_kind = theta_sfh['dm.kind'] # bins to rank order
    
        # scatter from noise -- subtract intrinsic assembly bias scatter from sig_SFMS 
        sig_noise = np.sqrt(Obvs.sigSSFR_SFMS(SHsnaps['m.star0'][indices])**2 - theta_sfh['sig_abias']**2)

        dsfr = np.repeat(-999., len(dMhalo))
        for nn in np.arange(21): 
            started = np.where(SHsnaps['nsnap_start'][indices] == nn)[0]

            m_kind_bin = np.arange(m_kind[started].min(), m_kind[started].max()+dm_kind, dm_kind)
            ibin = np.digitize(m_kind[started], m_kind_bin) 

            for i in np.unique(ibin): 
                inbin = np.where(ibin == i+1)

                isort = np.argsort(dMhalo[started[inbin]])

                dsfr[started[inbin]][isort] = np.sort(np.random.randn(len(inbin[0]))) * theta_sfh['sig_abias']

        assert dsfr.min() != -999.

        dsfr += sig_noise * np.random.randn(len(dMhalo))
        
        SHsnaps['sfr0'][indices] = mu_sfr + dsfr

        F_sfr = _logSFR_dSFR 

        sfr_kwargs = {'dSFR': dsfr, 'theta_sfms': theta_sfms}
    else:
        raise NotImplementedError
    return F_sfr, sfr_kwargs


def _logSFR_dSFR(logmm, zz, dSFR=None, theta_sfms=None): 
    return Obvs.SSFR_SFMS(logmm, zz, theta_SFMS=theta_sfms) + logmm + dSFR


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
