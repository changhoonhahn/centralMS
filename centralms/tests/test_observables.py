'''

Test for observables


'''
import numpy as np 
from letstalkaboutquench.fstarforms import fstarforms

import env 
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
    GroupCat_fSFMS()
