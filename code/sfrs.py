'''

Functions for handling star formation rates 


'''
import time
import numpy as np 

# --- local --- 
import util as UT


def LogSFR_sfms(logMstar, z_in, sfms_dict=None): 
    ''' Wrapper for SFMS star formation rates 
    '''
    if sfms_dict['name'] == 'constant_offset':  
        # the offset from the average SFMS is preserved throughout the redshift
        logsfr = AverageLogSFR_sfms(logMstar, z_in, sfms_dict=sfms_dict['sfms']) + \
                sfms_dict['dsfr']
    elif sfms_dict['name'] == 'no_scatter': 
        # SFR is just the average SFMS 
        logsfr = AverageLogSFR_sfms(logMstar, z_in, sfms_dict=sfms_dict['sfms'])
        
    elif sfms_dict['name'] == 'random_step':
        t = UT.t_from_z(z_in)
        ishift = np.abs(sfms_dict['tshift'] - 
                np.tile(t, (sfms_dict['tshift'].shape[1],1)).T).argmin(axis=1)
        ishift[np.where((sfms_dict['tshift'])[range(len(ishift)), ishift] > t)] -= 1
        dsfr = sfms_dict['amp'][range(len(ishift)), ishift]
        logsfr = AverageLogSFR_sfms(logMstar, z_in, sfms_dict=sfms_dict['sfms']) + dsfr

    return logsfr 


def LogSFR_Q(t, logSFR_Q=None, tau_Q=None, t_Q=None):
    ''' Wrapper for SFR of quenching galaxies
    '''
    SFRQ = np.power(10, logSFR_Q)
    logsfr = np.log10(SFRQ * np.exp( (t_Q - t) / tau_Q ) )
    return logsfr


def AverageLogSFR_sfms(mstar, z_in, sfms_dict=None): 
    ''' Model for the average SFR of the SFMS as a function of M* at redshift z_in.
    The model takes the functional form of 

    log(SFR) = A * log M* + B * z + C

    '''
    if sfms_dict is None: 
        raise ValueError

    if sfms_dict['name'] == 'linear': 
        # mass slope
        A_highmass = 0.53
        A_lowmass = 0.53
        try: 
            mslope = np.repeat(A_highmass, len(mstar))
        except TypeError: 
            mstar = np.array([mstar])
            mslope = np.repeat(A_highmass, len(mstar))
        # z slope
        zslope = sfms_dict['zslope']            # 0.76, 1.1
        # offset 
        offset = np.repeat(-0.11, len(mstar))

    elif sfms_dict['name'] == 'kinked': # Kinked SFMS 
        # mass slope
        A_highmass = 0.53 
        A_lowmass = sfms_dict['mslope_lowmass'] 
        try: 
            mslope = np.repeat(A_highmass, len(mstar))
        except TypeError: 
            mstar = np.array([mstar])
            mslope = np.repeat(A_highmass, len(mstar))
        lowmass = np.where(mstar < 9.5)
        mslope[lowmass] = A_lowmass
        # z slope
        zslope = sfms_dict['zslope']            # 0.76, 1.1
        # offset
        offset = np.repeat(-0.11, len(mstar))
        offset[lowmass] += A_lowmass - A_highmass 

    mu_SFR = mslope * (mstar - 10.5) + zslope * (z_in-0.0502) + offset
    return mu_SFR


def ScatterLogSFR_sfms(mstar, z_in, sfms_dict=None): 
    ''' Scatter of the SFMS logSFR as a function of M* and 
    redshift z_in. Hardcoded at 0.3 
    '''
    if sfms_dict is None: 
        raise ValueError
    return 0.3 


def integSFR(logsfr, mass0, t0, tf, mass_dict=None):
    ''' Integrated star formation rate stellar mass using Euler or RK4 integration

    M* = M*0 + f_retain * Int(SFR(t) dt, t0, tf)
    
    Parameters
    ----------
    logsfr : function 
        SFR function that accepts mass and t_cosmic as inputs 

    mass : ndarray
        initial stellar mass  

    t0 : ndarray 
        initial cosmic time 

    tf : ndarray
        final cosmic time

    f_retain : 
        fraction of stellar mass not lost from SNe and winds from Wetzel Paper
    '''
    type = mass_dict['type']            # Euler or RK4
    f_retain = mass_dict['f_retain']    # Retain fraction 
    delt = mass_dict['t_step']          # maximum t resolution of the integral 

    niter = int(np.ceil( (tf.max()-t0.min())/delt ))
    niters = np.ceil( (tf - t0) / delt).astype('int')  
    
    t_n_1 = t0 
    t_n = t_n_1 
    logSFR_n_1 = logsfr(mass0, t0)
    logM_n_1 = mass0
    #print niter, ' ', type, ' iterations'
    #print 'f_reatin = ', f_retain
    if niter > 0: 
        for i in xrange(niter): 
            iter_time = time.time()
            keep = np.where(niters > i) 
            t_n[keep] = t_n_1[keep] + delt
                
            if type == 'euler':         # Forward Euler Method
                logM_n_1[keep] = np.log10(
                        (10. ** logM_n_1[keep]) + 
                        delt * 10.**9. * f_retain * (10.** logSFR_n_1[keep])
                        )
            elif type == 'rk4':         # Runge Kutta 
                k1 = (10.0 ** logSFR_n_1)

                k2_sfr = logsfr(
                        np.log10(10.0**logM_n_1 + (10**9 * delt)/2.0 * k1), 
                        t_n_1 + delt/2.0)
                k2 = (10.0 ** k2_sfr)

                k3_sfr = logsfr(
                        np.log10(10.0**logM_n_1 + (10**9 * delt)/2.0 * k2), 
                        t_n_1 + delt/2.0)
                k3 = (10.0 ** k3_sfr)

                k4_sfr = logsfr(
                        np.log10(10.0**logM_n_1 + (10**9 * delt) * k3), 
                        t_n_1 + delt)
                k4 = (10.0 ** k4_sfr)

                logM_n_1[keep] = np.log10(10.0 ** logM_n_1[keep] + f_retain/6.0 * (delt * 10**9) * (k1[keep] + 2.0*k2[keep] + 2.0*k3[keep] + k4[keep])) 

            else: 
                raise NotImplementedError
            
            if np.sum(np.isnan(logM_n_1)) > 0: 
                raise ValueError('There are NaNs') 
    
            # update log(SFR), and t from step n-1
            logSFR_n_1[keep] = logsfr(logM_n_1, t_n)[keep]
            t_n_1 = t_n

    # sanity check
    if np.min(logM_n_1 - mass0) < 0.0: 
        if np.min(logM_n_1 - mass0) > -0.001: 
            pass
        else: 
            raise ValueError("integrated mass cannot decrease over cosmic time")

    return logM_n_1, logSFR_n_1


def ODE_Euler(dydt, init_cond, t_arr, delt, **func_args): 
    '''
    '''
    # t where we will evaluate 
    t_eval = np.arange(t_arr.min(), t_arr.max()+delt, delt) 
    t_eval[-1] = t_arr[-1]

    indices = []
    for tt in t_arr[1:-1]:
        idx = np.argmin(np.abs(t_eval - tt))
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

    indices = []
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


def dlogMdt_MS(logMstar, t, t_initial=None, t_final=None, f_retain=None, zfromt=None, sfh_kwargs=None): 
    ''' Integrand d(logM)/dt for solving the ODE 

    d(logM)/dt = SFR'(logM, t) * 10^9/(M ln(10))

    SFR'(t) = SFR(M*, t+t_offset) 
    or  
            = 0 if t > tf - t_offset
    '''
    dlogMdt = np.zeros(len(logMstar))
    within = np.where((t <= t_final) & (t >= t_initial) )
    if len(within[0]) > 0:  
        try: 
            dsfr = dSFR_MS(t, sfh_kwargs)[within]
        except TypeError:
            dsfr = dSFR_MS(t, sfh_kwargs)

        tmp = AverageLogSFR_sfms(
                logMstar[within], 
                zfromt(t), 
                sfms_dict=sfh_kwargs['sfms']) + dsfr + \
                        9. - \
                        logMstar[within] + \
                        np.log10(f_retain) - \
                        0.3622157
        dlogMdt[within] = np.power(10, tmp)

    return dlogMdt 


def dSFR_MS(t, sfh_kwargs): 
    '''
    '''
    if sfh_kwargs['name'] == 'constant_offset':  
        dsfr = sfh_kwargs['dsfr']
    elif sfh_kwargs['name'] == 'no_scatter': 
        dsfr = 0.
    elif sfh_kwargs['name'] == 'random_step':
        ishift = np.abs(sfh_kwargs['tshift'] - t).argmin(axis=1)
        ishift[np.where((sfh_kwargs['tshift'])[range(len(ishift)), ishift] > t)] -= 1
        dsfr = sfh_kwargs['amp'][range(len(ishift)), ishift]
    return dsfr


def dlogMdt_Q(logMstar, t, logSFR_Q=None, tau_Q=None, t_Q=None, f_retain=None, t_final=None): 
    ''' dlogM/dt for quenching galaxies. Note that this is derived from dM/dt.  

    dlogM/dt quenching = SFR(M_Q, t_Q)/(M ln10) * exp( (t_Q - t) / tau_Q ) 
    '''
    dlogMdt = np.zeros(len(logMstar))

    within = np.where((t <= t_final) & (t >= t_Q))
    if len(within[0]) > 0:  
        SFRQ = np.power(10, logSFR_Q[within] + 9. - logMstar[within])

        dlogMdt[within] = f_retain * SFRQ * \
                np.exp( (t_Q[within] - t) / tau_Q[within] ) / np.log(10)  
    return dlogMdt 


def logSFRt_MS(mstar, t, method_kwargs=None):
    ''' log SFR(t) for different methods 
    '''
    if method_kwargs['name'] == 'constant_offset':  
        # the offset from the average SFMS is preserved throughout the redshift
        mu_logsfr = AverageLogSFR_sfms(mstar, UT.z_from_t(t), sfms_dict=method_kwargs['sfms'])
        return mu_logsfr + method_kwargs['dsfr']

    elif method_kwargs['name'] == 'no_scatter': 
        # SFR is just the average SFMS 
        mu_logsfr = AverageLogSFR_sfms(mstar, UT.z_from_t(t), sfms_dict=method_kwargs['sfms'])
        return mu_logsfr


def logSFRt_Q(MQ, t, tQ=None, tau_dict=None, method_kwargs=None): 
    ''' log SFR(t) after tQ to tf for quenching galaxies (NOTE THAT THIS IS VERY SPECIFIC)
                                
    log(SFR)_quenching = np.log10( np.exp( -(t_f - t_Q)/tau) )
    '''
    if method_kwargs == 'constant_offset': 
        mu_logsfr = AverageLogSFR_sfms(MQ, UT.z_from_t(tQ), sfms_dict=method_kwargs['sfms'])

        tauQ = getTauQ(MQ, tau_dict=tau_dict)
        
        dlogsfrq = np.log10( np.exp( (tQ - t) / tauQ ) ) 
        
        return mu_logsfr + method_kwargs['dsfr'] + dlogsfrq  

    elif method_kwargs == 'no_scatter':
        mu_logsfr = AverageLogSFR_sfms(MQ, UT.z_from_t(tQ), sfms_dict=method_kwargs['sfms'])

        tauQ = getTauQ(MQ, tau_dict=tau_dict)
        dlogsfrq = np.log10( np.exp( (tQ - t) / tauQ ) ) 
        
        return mu_logsfr + dlogsfrq  


def getTauQ(mstar, tau_dict=None): 
    ''' Return quenching efold based on stellar mass of galaxy, Tau(M*).
    '''
    type = tau_dict['name']
    if type == 'constant':      # constant tau 

        n_arr = len(mstar) 
        tau = np.array([0.5 for i in xrange(n_arr)]) 

    elif type == 'linear':      # lienar tau(mass) 

        tau = -(0.8 / 1.67) * ( mstar - 9.5) + 1.0
        #if np.min(tau) < 0.1: #    tau[ tau < 0.1 ] = 0.1
         
    elif type == 'instant':     # instant quenching 

        n_arr = len(mstar) 
        tau = np.array([0.001 for i in range(n_arr)]) 

    elif type == 'discrete': 
        # param will give 4 discrete tau at the center of mass bins 
        masses = np.array([9.75, 10.25, 10.75, 11.25]) 

        if param is None: 
            raise ValueError('asdfasdfa') 

        tau = np.interp(mstar, masses, param) 
        tau[ tau < 0.05 ] = 0.05

    elif type == 'line': 
        # param will give slope and yint of pivoted tau line 
        
        tau = tau_dict['slope'] * (mstar - tau_dict['fid_mass']) + tau_dict['yint']
        
        try: 
            if np.min(tau) < 0.001: 
                tau[np.where( tau < 0.001 )] = 0.001
        except ValueError: 
            pass 

    elif type == 'satellite':   # quenching e-fold of satellite

        tau = -0.57 * ( mstar - 9.78) + 0.8
        if np.min(tau) < 0.001:     
            tau[np.where( tau < 0.001 )] = 0.001

    elif type == 'long':      # long quenching (for qa purposes)

        n_arr = len(mstar) 
        tau = np.array([2.0 for i in xrange(n_arr)]) 

    else: 
        raise NotImplementedError('asdf')

    return tau 
