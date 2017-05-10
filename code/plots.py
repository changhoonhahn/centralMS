'''

Key plots to illustrate results 


'''
import numpy as np 
import pickle

import catalog as Cat
import evolver as Evol
import observables as Obvs
import util as UT

import matplotlib.pyplot as plt 
import bovy_plot as bovy
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


def siglogMstar_tduty(nsnap0): 
    ''' Plot sigma_logM*(M_halo = 10^12) as a function of t_duty (the duty cycle
    timescale) 
    '''
    fig = plt.figure()
    sub = fig.add_subplot(111)

    fig_data = []  
    tdutys = np.array([0.1, 0.5, 1., 5., 10.])

    for sfh in ['random_step_fluct']: # different SFH prescriptions
        sigmaMstar = np.repeat(-999., len(tdutys))

        for i_t, tduty in enumerate(tdutys):  # t_duty 
            # load in subhalo catalog
            subhist = Cat.PureCentralHistory(nsnap_ancestor=nsnap0)
            subcat = subhist.Read()

            # load in generic theta (parameter values)
            theta = Evol.defaultTheta(sfh) 
            theta['sfh']['dt_min'] = tduty
            theta['sfh']['dt_max'] = tduty
            theta['mass']['t_step'] = 0.05
            if theta['mass']['t_step'] > tduty/10.: 
                theta['mass']['t_step'] = tduty/10.
            
            eev = Evol.Evolver(subcat, theta, nsnap0=nsnap0)
            eev.Initiate()
            eev.Evolve() 
            
            # galaxy/subhalo catalog
            subcat = eev.SH_catalog
    
            isSF = np.where(subcat['gclass'] == 'star-forming')[0]
    
            smhmr = Obvs.Smhmr()
            m_mid, mu_mstar, sig_mstar, cnts = smhmr.Calculate(subcat['halo.m'][isSF], subcat['m.star'][isSF])
            sig_mstar_mh12 = smhmr.sigma_logMstar(subcat['halo.m'][isSF], subcat['m.star'][isSF], Mhalo=12.)
            sigmaMstar[i_t] = sig_mstar_mh12

            # save data  
            fig_datum = {
                    'sfh': sfh, 
                    'tduty': tduty,
                    'theta': theta,
                    'smhmr': [m_mid, mu_mstar, sig_mstar, cnts], 
                    'sigmaM*(Mh12)': sig_mstar_mh12
                    }
            fig_data.append(fig_datum)

        sub.scatter(tdutys, sigmaMstar, lw=0, s=15, label=sfh)

    sub.plot([0., 11.], [0.2, 0.2], lw=3, ls='--', c='k', label='Obsv.') 

    sub.set_xlim([0., 11.])
    sub.set_ylim([0., 0.5])
    fig.savefig(''.join([UT.fig_dir(), 'sigmalogMstar_tduty.png']), bbox_inches='tight')

    # dump to pickle file 
    pickle.dump(fig_data, open(''.join([UT.dat_dir(), 'fig_data/sigmalogMstar_tduty.p']), 'wb'))
    return None 


if __name__=='__main__': 
    siglogMstar_tduty(15)
