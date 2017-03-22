'''





'''
import numpy as np 
from scipy.interpolate import interp1d
import observables as Obvs



class Evolver(object): 
    def __init__(self, PCH_catalog, theta_init, theta_evol, nsnap_i=20): 
        '''
        '''
        pass




def assignSFRs(masses, zs, theta_GV=None, theta_SFMS=None, theta_FQ=None): 
    ''' Given stellar masses, zs, and parameters that describe the 
    green valley, SFMS, and FQ return SFRs

    Details: 
    -------
    - Designates a fraction of galaxies as green valley "quenching" galaxies based on theta_GV

    
    Parameters
    ----------
    masses : (array)
        Array that of stellar masses
    
    zs : (array)
        Array that of stellar zs 

    theta_XX : (dict) 
        Dictary that specifies XX property  

    Return: 
    ------
    output : (dict) 
        Dictionary that specifies the following 
    '''
    np.random.seed()
    qf = Obvs.Fq()   # initialize quiescent fraction class 

    # check inputs 
    if theta_GV is None: 
        raise ValueError("Specify green valley parameters")
    if theta_SFMS is None: 
        raise ValueError("Specify Star-Forming Main Sequence parameters")
    if theta_FQ is None: 
        raise ValueError("Specify Quiescent Fraction parameters")

    assert len(masses) > 0  
    assert len(masses) == len(zs) 

    ngal = len(masses)   # N_gals

    # set up output 
    output = {} 
    for key in ['SFR', 'Gclass', 'MQ']: 
        if key != 'Gclass':  
            output[key] = np.repeat(-999., ngal)
        else: 
            output[key] = np.repeat('', ngal).astype('|S16') 
    
    # Assign Green valley galaxies 
    f_gv = lambda mm: theta_GV['slope'] * (mm - theta_GV['fidmass']) + theta_GV['offset'] # f_GV 
    
    rand = np.random.uniform(0., 1., ngal)
    isgreen = np.where(rand < f_gv(masses))
    output['Gclass'][isgreen] = 'qing'
    output['MQ'][isgreen] = masses[isgreen]                     # M* at quenching
    output['SFR'][isgreen] = np.random.uniform(                 # sample SSFR from uniform distribution 
            SSFR_Qpeak(masses[isgreen]),                        # between SSFR_Qpeak
            SSFR_SFMS(masses[isgreen], zs[isgreen]),     # and SSFR_SFMS 
            len(isgreen[0])) + masses[isgreen]

    # GV galaxy queiscent fraction 
    gv_fQ = qf.Calculate(mass=masses[isgreen], sfr=output['SFR'][isgreen], z=zs[isgreen], theta_SFMS=theta_SFMS)
    
    # quiescent galaxies 
    fQ_gv = interp1d(gv_fQ[0], gv_fQ[1] * f_gv(gv_fQ[0]))
    
    isnotgreen = np.where(rand >= f_gv(masses))[0]
    fQ_true = qf.model(masses[isnotgreen], zs[isnotgreen], lit=theta_FQ['name']) - fQ_gv(masses[isnotgreen])
    isq = isnotgreen[np.where(rand[isnotgreen] < fQ_true)]
    Nq = len(isq)

    output['Gclass'][isq] = 'quiescent'
    output['SFR'][isq] = SSFR_Qpeak(masses[isq]) + np.random.randn(Nq) * sigSSFR_Qpeak(masses[isq]) + masses[isq]
    
    # star-forming galaxies 
    issf = np.where(output['Gclass'] == '')
    Nsf = len(issf[0])

    output['Gclass'][issf] = 'starforming'
    output['SFR'][issf] = SSFR_SFMS(masses[issf], zs[issf], theta_SFMS=theta_SFMS) + \
            np.random.randn(Nsf) * sigSSFR_SFMS(masses[issf]) + \
            masses[issf]
    
    return output

    
def SSFR_Qpeak(mstar):  
    ''' Roughly the average of the log(SSFR) of the quiescent peak 
    of the SSFR distribution. This is designed to reproduce the 
    Brinchmann et al. (2004) SSFR limits.
    '''
    #return -0.4 * (mstar - 11.1) - 12.61
    return 0.4 * (mstar - 10.5) - 1.73 - mstar 


def sigSSFR_Qpeak(mstar):  
    ''' Scatter of the log(SSFR) quiescent peak of the SSFR distribution 
    '''
    return 0.18 


def SSFR_SFMS(mstar, z_in, theta_SFMS=None): 
    ''' Model for the average SSFR of the SFMS as a function of M* at redshift z_in.
    The model takes the functional form of 

    log(SFR) = A * log M* + B * z + C

    '''
    assert theta_SFMS is not None 

    if theta_SFMS['name'] == 'linear': 
        # mass slope
        A_highmass = 0.53
        A_lowmass = 0.53
        try: 
            mslope = np.repeat(A_highmass, len(mstar))
        except TypeError: 
            mstar = np.array([mstar])
            mslope = np.repeat(A_highmass, len(mstar))
        # z slope
        zslope = theta_SFMS['zslope']            # 0.76, 1.1
        # offset 
        offset = np.repeat(-0.11, len(mstar))

    elif theta_SFMS['name'] == 'kinked': # Kinked SFMS 
        # mass slope
        A_highmass = 0.53 
        A_lowmass = theta_SFMS['mslope_lowmass'] 
        try: 
            mslope = np.repeat(A_highmass, len(mstar))
        except TypeError: 
            mstar = np.array([mstar])
            mslope = np.repeat(A_highmass, len(mstar))
        lowmass = np.where(mstar < 9.5)
        mslope[lowmass] = A_lowmass
        # z slope
        zslope = theta_SFMS['zslope']            # 0.76, 1.1
        # offset
        offset = np.repeat(-0.11, len(mstar))
        offset[lowmass] += A_lowmass - A_highmass 

    mu_SSFR = (mslope * (mstar - 10.5) + zslope * (z_in-0.0502) + offset) - mstar
    return mu_SSFR


def sigSSFR_SFMS(mstar): #, z_in, theta_SFMS=None): 
    ''' Scatter of the SFMS logSFR as a function of M* and 
    redshift z_in. Hardcoded at 0.3 
    '''
    assert theta_SFMS is not None
    return 0.3 


def defaultTheta(): 
    ''' Return the most vanilla default parameter values  
    '''
    #####
    #####
    #####
    #####
    #####
    #####
    #####
    return theta 
