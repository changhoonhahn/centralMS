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
    pretty_colors = prettycolors()
    fig = plt.figure()
    sub = fig.add_subplot(111)

    fig_data = []  
    tdutys = np.array([0.1, 0.5, 1., 5., 10.])
    
    for i_a, abias in enumerate([0., 0.1, 0.2, 0.3]): 
        if abias == 0.: 
            sfh = 'random_step_fluct'
        else: 
            sfh = 'random_step_abias2'
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
            if abias > 0.: 
                theta['sfh']['t_abias'] = 2.  # assembly bias timescale
                theta['sfh']['sigma_corr'] = abias
            else: 
                theta['sfh']['t_abias'] = None   # assembly bias timescale
            
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
                    'sigma_corr': abias, 
                    't_abias': theta['sfh']['t_abias'],
                    'theta': theta,
                    'smhmr': [m_mid, mu_mstar, sig_mstar, cnts], 
                    'sigmaM*(Mh12)': sig_mstar_mh12
                    }
            fig_data.append(fig_datum)
        
        sfh_label = ''.join(['$\sigma_{corr} = ', str(abias), ', ', 
            't_{abias} = ', str(theta['sfh']['t_abias']), '$']) 
        sub.scatter(tdutys, sigmaMstar, lw=0, s=20, c=pretty_colors[i_a], label=sfh_label)
        sub.plot(tdutys, sigmaMstar, lw=2, c=pretty_colors[i_a])

    sub.plot([0., 11.], [0.2, 0.2], lw=3, ls='--', c='k', label='Obsv.') 

    sub.legend(loc='lower right')

    sub.set_xlim([0., 11.])
    sub.set_ylim([0., 0.6])
    fig.savefig(''.join([UT.fig_dir(), 'sigmalogMstar_tduty.png']), bbox_inches='tight')

    out_file = ''.join([UT.dat_dir(), 
        'fig_data/', 
        'sigmalogMstar', 
        '.tduty', '_'.join(tdutys.astype('str'))]) 
    # dump to pickle file 
    pickle.dump(fig_data, open(out_file+'.p', 'wb'))
    
    f = open(out_file+'.dat', 'w')
    f.write('# SFH, t_duty [Gyr], sigma_corr, t_abias, SMHMR sigma logMstar at Mhalo=10^12 \n')
    for datum in fig_data: 
        out_line = '\t'.join([datum['sfh'], str(datum['tduty']), str(datum['sigma_corr']), 
            str(datum['t_abias']), str(datum['sigmaM*(Mh12)'])])+'\n'
        f.write(out_line)
    f.close()

    return None 


if __name__=='__main__': 
    siglogMstar_tduty(15)
