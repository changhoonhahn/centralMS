'''

Test for observables


'''
import numpy as np 
from letstalkaboutquench.fstarforms import fstarforms

import env 
import util as UT
import catalog as Cat
import observables as Obvs

import matplotlib as mpl 
import matplotlib.pyplot as plt 
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5


def SFMS_z1(): 
    ''' Comparison of different SFMS observations at z~1.
    '''
    # Lee et al. (2015)
    logSFR_lee = lambda mm: 1.53 - np.log10(1 + (10**mm/10**(10.10))**-1.26)
    # Noeske et al. (2007) 0.85 < z< 1.10 (by eye)
    logSFR_noeske = lambda mm: (1.580 - 1.064)/(11.229 - 10.029)*(mm - 10.0285) + 1.0637
    # Moustakas et al. (2013) 0.8 < z < 1. (by eye)  
    logSFR_primus = lambda mm: (1.3320 - 1.296)/(10.49 - 9.555) * (mm-9.555) + 1.297
    
    # prior range
    logSFR_prior_min = lambda mm: (0.6 * (mm - 10.5) + 0.9 * (1.-0.0502) - 0.11) 
    logSFR_prior_max = lambda mm: (0.6 * (mm - 10.5) + 2. * (1.-0.0502) - 0.11) 


    fig = plt.figure(1)
    sub = fig.add_subplot(111)
    marr = np.linspace(9., 12., 100)
    sub.plot(marr, logSFR_lee(marr), label='Lee et al.(2015)')
    sub.plot(marr, logSFR_noeske(marr), label='Noeske et al.(2007)')
    sub.plot(marr, logSFR_primus(marr), label='Moustakas et al.(2013)')
    sub.fill_between(marr, logSFR_prior_min(marr), logSFR_prior_max(marr), label='Prior', alpha=0.5)
    sub.legend(loc='lower right') 
    sub.set_xlim([9.5, 11.5])
    fig.savefig(UT.fig_dir()+'test.z_1.sfms.png')
    plt.close() 
    return None


def GroupCat_fSFMS(): 
    ''' 
    '''
    
    real = Cat.Observations('group_catalog', Mrcut=18, position='central')
    catalog = real.Read()
    # calculate the quenched fraction 
    qf = Obvs.Fq()
    qfrac = qf.Calculate(mass=catalog['mass'], sfr=catalog['sfr'], z=0.05, theta_SFMS={'name': 'linear', 'zslope': 1.05, 'mslope':0.53})
    
    # get SFMS frac fit from LetsTalkAboutQuench 
    fSFMS = fstarforms() 
    _ = fSFMS.fit(catalog['mass'], catalog['sfr'], method='gaussfit')
    fit_logm, fit_fsfms = fSFMS.frac_SFMS()

    fig = plt.figure(1)
    sub = fig.add_subplot(111)
    sub.plot(qfrac[0], qfrac[1], label='quiescent fraction')
    sub.plot(fit_logm, 1.-fit_fsfms, label='1-$f_\mathrm{SFMS}$') 
    sub.set_xlim([9.5, 12.])
    sub.set_xlabel('log $M_*\;\;[M_\odot]$', fontsize=20)
    sub.set_ylim([0., 1.])
    sub.set_ylabel('$f_\mathrm{Q}$', fontsize=20)
    plt.show()
    plt.close()
    return None


if __name__=='__main__': 
    SFMS_z1()
    #GroupCat_fSFMS()
