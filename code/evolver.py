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
    output['MQ'][isgreen] = masses[isgreen]         # M* at quenching
    output['SFR'][isgreen] = np.random.uniform(     # sample SSFR from uniform distribution 
            Obvs.SSFR_Qpeak(masses[isgreen]),       # between SSFR_Qpeak
            Obvs.SSFR_SFMS(masses[isgreen], zs[isgreen], theta_SFMS=theta_SFMS),                
            len(isgreen[0])) + masses[isgreen]      # and SSFR_SFMS 

    # GV galaxy queiscent fraction 
    gv_fQ = qf.Calculate(mass=masses[isgreen], sfr=output['SFR'][isgreen], z=zs[isgreen], 
            mass_bins=np.arange(7.8, 12.2, 0.2), theta_SFMS=theta_SFMS)
    
    # quiescent galaxies 
    fQ_gv = interp1d(gv_fQ[0], gv_fQ[1] * f_gv(gv_fQ[0]))
    
    isnotgreen = np.where(rand >= f_gv(masses))[0]
    fQ_true = qf.model(masses[isnotgreen], zs[isnotgreen], lit=theta_FQ['name']) - fQ_gv(masses[isnotgreen])
    isq = isnotgreen[np.where(rand[isnotgreen] < fQ_true)]
    Nq = len(isq)

    output['Gclass'][isq] = 'quiescent'
    output['SFR'][isq] = Obvs.SSFR_Qpeak(masses[isq]) + \
            np.random.randn(Nq) * Obvs.sigSSFR_Qpeak(masses[isq]) + masses[isq]
    
    # star-forming galaxies 
    issf = np.where(output['Gclass'] == '')
    Nsf = len(issf[0])

    output['Gclass'][issf] = 'starforming'
    output['SFR'][issf] = Obvs.SSFR_SFMS(masses[issf], zs[issf], theta_SFMS=theta_SFMS) + \
            np.random.randn(Nsf) * Obvs.sigSSFR_SFMS(masses[issf]) + \
            masses[issf]
    
    return output

    
def defaultTheta(): 
    ''' Return generic default parameter values
    '''
    theta = {} 

    theta['gv'] = {'slope': 1.03, 'fidmass': 10.5, 'offset': -0.02}
    theta['sfms'] = {'name': 'linear', 'zslope': 1.14}
    theta['fq'] = {'name': 'cosmos_tinker'}

    return theta 
