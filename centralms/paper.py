''' 

Make figures for paper 


'''
import numpy as np 
from scipy.interpolate import interp1d

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
    subcat, eev = ABC.model('randomSFH', np.array([1.35, 0.6]), nsnap0=nsnap0, 
            downsampled='14', sigma_smhm=0.2, forTests=True)

    # randomly pick a galaxy that match the below criteria
    isSF = np.where(subcat['gclass'] == 'sf') 
    eligible = np.where((subcat['nsnap_start'][isSF] == nsnap0) & 
            (subcat['weights'][isSF] > 0.) & 
            (subcat['m.star0'][isSF] > 10.) & 
            (subcat['m.star0'][isSF] < 10.25))
    i_random = np.random.choice(eligible[0], size=1)
    i_gal = isSF[0][i_random]

    # track back it's M* and SFR
    mstar_hist, sfr_hist = [subcat['m.star0'][i_gal][0]], [subcat['sfr0'][i_gal][0]] 
    
    for isnap in range(2,nsnap0)[::-1]: 
        mstar_hist.append(subcat['m.star.snap'+str(isnap)][i_gal][0])
        sfr_hist.append(subcat['sfr.snap'+str(isnap)][i_gal][0])
    mstar_hist.append(subcat['m.star'][i_gal][0]) 
    sfr_hist.append(subcat['sfr'][i_gal][0]) 
    sfr_hist = np.array(sfr_hist)
    
    # SFMS 
    sfr_sfms = [Obvs.SSFR_SFMS(mstar_hist[0], UT.z_nsnap(nsnap0), 
        theta_SFMS=subcat['theta_sfms']) + mstar_hist[0]]
    for ii, isnap in enumerate(range(2,nsnap0)[::-1]): 
        sfr_sfms.append(Obvs.SSFR_SFMS(mstar_hist[ii+1], UT.z_nsnap(isnap), 
            theta_SFMS=subcat['theta_sfms']) + mstar_hist[ii+1])
    sfr_sfms.append(Obvs.SSFR_SFMS(mstar_hist[-1], UT.z_nsnap(1), 
            theta_SFMS=subcat['theta_sfms']) + mstar_hist[-1])
    sfr_sfms = np.array(sfr_sfms)[:,0]
    f_sfms = interp1d(mstar_hist, sfr_sfms) #smooth

    pretty_colors = prettycolors()  
    fig = plt.figure(figsize=(15,8))
    # log SFR - log M* galaxy evolution 
    sub1 = fig.add_subplot(121)
    marr = np.linspace(mstar_hist[0], mstar_hist[-1], 100)
    sub1.plot(marr, f_sfms(marr, c='k', lw=2)
    sub1.plot(mstar_hist, sfr_hist, c=pretty_colors[2])
    #sub1.plot(mstar_hist, sfr_sfms, c='k', lw=2)
    # plot SFMS(M*, z)
    m_arr = np.linspace(9., 12., 100)
    for z in np.arange(0.1, 1.25, 0.25): 
        sub1.plot(m_arr, Obvs.SSFR_SFMS(m_arr, z, theta_SFMS=subcat['theta_sfms']) + m_arr,
                c='k', ls='--', lw=0.5)
    sub1.set_xlim([10., 11.])
    sub1.set_xlabel('$\mathtt{log(\; M_*\; [M_\odot]\;)}$', fontsize=25)
    sub1.set_ylim([-1., 2.])
    sub1.set_ylabel('$\mathtt{log(\; SFR\; [M_\odot/yr]\;)}$', fontsize=25)
    
    # Delta log SFR(t) evolution 
    sub2 = fig.add_subplot(122)
    xx, yy = [], []
    for i in range(len(eev.dlogSFR_amp[i_random][0])-1):
        xx.append(eev.tsteps[i_random][0][i]) 
        yy.append(eev.dlogSFR_amp[i_random][0][i])
        xx.append(eev.tsteps[i_random][0][i+1]) 
        yy.append(eev.dlogSFR_amp[i_random][0][i])
    sub2.plot([UT.t_nsnap(nsnap0), UT.t_nsnap(1)], [0.,0.], ls='--', c='k')
    sub2.plot(xx, yy)
    sub2.set_xlim([UT.t_nsnap(nsnap0), UT.t_nsnap(1)]) 
    sub2.set_xlabel('$\mathtt{t_{cosmic}\;[Gyr]}$', fontsize=25)
    sub2.set_ylim([-1., 1.]) 
    sub2.set_ylabel('$\Delta \mathtt{log(\;SFR\;[M_\odot/yr])}$', fontsize=25)
    fig.savefig(''.join([UT.fig_dir(), 'paper1.png']), bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    SFHmodel()
