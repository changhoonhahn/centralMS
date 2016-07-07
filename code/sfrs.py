'''

Functions for handling star formation rates 


'''
import time
import numpy as np 

# --- local --- 
import util as UT


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


def SFRt_MS_nothing(mstar, t, dsfr, sfms_dict=None):
    ''' log SFR(t) for the `nothing` scenario
    '''
    mu_logsfr = AverageLogSFR_sfms(mstar, UT.z_from_t(t), sfms_dict=sfms_dict)
    return mu_logsfr + dsfr


def SFRt_Q_nothing(MQ, t, dsfr, tQ=None, sfms_dict=None, tau_dict=None): 
    ''' log SFR(t) after tQ to tf for quenching galaxies (NOTE THAT THIS IS VERY SPECIFIC)
                                
    log(SFR)_quenching = np.log10( np.exp( -(t_f - t_Q)/tau) )
    '''
    mu_logsfr = AverageLogSFR_sfms(MQ, UT.z_from_t(tQ), sfms_dict=sfms_dict)

    tauQ = getTauQ(MQ, tau_dict=tau_dict)
    
    dlogsfrq = np.log10( np.exp( (tQ - t) / tauQ ) ) 
    
    return mu_logsfr + dsfr + dlogsfrq  


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
