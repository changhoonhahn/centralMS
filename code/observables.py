'''


Functions to measure observables of a given galaxy sample. 
The main observables are: 
    SFMS, SFM, SMHMR


'''
import numpy as np 

# --- local --- 
import centralms as CMS 
from sham_hack import SMFClass 


def getSMF(masses, m_arr=None, dlogm=0.1, box=250, h=0.7): 
    ''' Calculate the Stellar Mass Function for a given set of stellar masses.
    '''
    if m_arr is None: 
        m_arr = np.arange(6.0, 12.1, dlogm) 

    vol = box ** 3  # box volume
    
    Ngal, mbin_edges = np.histogram(masses, bins=m_arr) # number of galaxies in mass bin  

    mbin = 0.5 * (mbin_edges[:-1] + mbin_edges[1:]) 
    phi = Ngal.astype('float') / vol /dlogm * h**3

    return [mbin, phi]


def analyticSMF(redshift, m_arr=None, dlogm=0.1, source='li-drory-march'): 
    ''' Analytic SMF for a given redshift. 

    Return
    ------
    [masses, phi] : 
        array of masses, array of phi (smf number density) 
    '''
    if redshift < 0.1:
        redshift = 0.1
    if m_arr is None: 
        m_arr = np.arange(6.0, 12.1, dlogm) 

    MF = SMFClass(source=source, redshift=redshift)
    
    mass, phi = [], [] 
    for mi in m_arr: 
        if source in ('cool_ages', 'blanton'): 
            mass.append(mi - 0.5 * dlogm)
            phi.append(MF.numden(-mi, -mi+dlogm)/dlogm) 
        else: 
            mass.append(mi + 0.5 * dlogm)
            phi.append(MF.numden(mi, mi+dlogm)/dlogm) 
    #print 'Analytic ', np.sum(np.array(phi))
    return [np.array(mass), np.array(phi)]
