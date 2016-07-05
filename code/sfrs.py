'''

Functions for handling star formation rates 


'''
import numpy as np 



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
