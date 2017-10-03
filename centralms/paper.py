''' 

Make figures for paper 


'''
import numpy as np 

import util as UT
import abcee as ABC
import observables as Obvs

import matplotlib.pyplot as plt 
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


def SFHmodel(nsnap0=15):
    ''' Figure that illustrates the SFH of galaxies. 
    Two panel plot. Panel a) SFH of a galaxy plotted alongside SFMS 
    '''
    subcat = ABC.model('randomSFH', np.array([1.35, 0.6]), nsnap0=nsnap0, 
            downsampled='14', sigma_smhm=0.2)

    # randomly pick a galaxy that match the below criteria
    eligible = np.where((subcat['nsnap_start'] == nsnap0) & 
            (subcat['weights'] > 0.) & 
            (subcat['gclass'] == 'sf') & 
            (subcat['m.star0'] > 9.5) & 
            (subcat['m.star0'] < 9.75))
    i_gal = np.random.choice(eligible[0], size=1)

    # track back it's M* and SFR
    mstar_hist, sfr_hist = [subcat['m.star0'][i_gal][0]], [subcat['sfr0'][i_gal][0]] 
    
    for isnap in range(2,nsnap0)[::-1]: 
        mstar_hist.append(subcat['m.star.snap'+str(isnap)][i_gal][0])
        sfr_hist.append(subcat['sfr.snap'+str(isnap)][i_gal][0])
    mstar_hist.append(subcat['m.star'][i_gal][0]) 
    sfr_hist.append(subcat['sfr'][i_gal][0]) 
    
    # SFMS 
    sfr_sfms = [Obvs.SSFR_SFMS(mstar_hist[0], UT.z_nsnap(nsnap0), 
        theta_SFMS=subcat['theta_sfms']) + mstar_hist[0]]
    for ii, isnap in enumerate(range(2,nsnap0)[::-1]): 
        sfr_sfms.append(Obvs.SSFR_SFMS(mstar_hist[ii+1], UT.z_nsnap(isnap), 
            theta_SFMS=subcat['theta_sfms']) + mstar_hist[ii+1])
    sfr_sfms.append(Obvs.SSFR_SFMS(mstar_hist[-1], UT.z_nsnap(1), 
            theta_SFMS=subcat['theta_sfms']) + mstar_hist[-1]
    
    fig = plt.figure(figsize=(15,8))
    sub1 = fig.add_subplot(211)
    sub1.scatter(mstar_hist, sfr_hist)
    sub1.plot(mstar_hist, sfr_sfms, c='k', lw=3)
    sub1.set_xlim([9., 12.])
    sub1.set_xlabel('$\mathtt{log(\; M_*\; [M_\odot]\;)}$', fontsize=25)
    sub1.set_ylim([-4., 2.])
    sub1.set_ylabel('$\mathtt{log(\; SFR\; [M_\odot/yr]\;)}$', fontsize=25)
    plt.show() 
    return None 


if __name__=="__main__": 
    SFHmodel()
