'''


Investigate the accretion history


'''
import numpy as np 
import matplotlib.pyplot as plt
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors 

# --- local ---
import util as UT 
import sfrs as SFR
import centralms as CMS 


def SHAM_AccHistory(): 
    ''' Investigate the accretion histories from SHAM 
    '''
    # Read CenQue object 
    galpop = CMS.Read_CenQue('sfms', cenque='default')

    Mhalo_hist = galpop.Mhalo_hist
    Msham_hist = galpop.Msham_hist
    
    z_acc, t_acc = UT.zt_table()
    t_cosmic  = t_acc[1:16]

    m_bin = np.where((galpop.halo_mass > 12.5) & (galpop.halo_mass < 13.)) 

    Mhalo_hist_bin = Mhalo_hist[m_bin, :][0]
    frac_Mhalo_hist = 10.**(
            Mhalo_hist_bin - 
            np.tile(Mhalo_hist_bin[:,0], Mhalo_hist_bin.shape[1]).reshape(
                Mhalo_hist_bin.shape[1], Mhalo_hist_bin.shape[0]).T
            )

    Msham_hist_bin = Msham_hist[m_bin, :][0]
    frac_Msham_hist = 10.**(
            Msham_hist_bin - 
            np.tile(Msham_hist_bin[:,0], Msham_hist_bin.shape[1]).reshape(
                Msham_hist_bin.shape[1], Msham_hist_bin.shape[0]).T
            )
    #frac_Msham_hist = 10.**(
    #        Msham_hist[m_bin, :] - msham_avg)[0]
    
    prettyplot() 
    pretty_colors = prettycolors() 
    fig = plt.figure()
    halo_sub = fig.add_subplot(121)
    sham_sub = fig.add_subplot(122)
    
    halo_percent = np.zeros((len(t_cosmic), 5))
    sham_percent = np.zeros((len(t_cosmic), 5))
    for i in xrange(Mhalo_hist.shape[1]): 
        rm_NaN = np.where(Mhalo_hist_bin[:,i] != -999.) 
        a, b, c, d, e = np.percentile(frac_Mhalo_hist[rm_NaN,i], [2.5, 16, 50, 84, 97.5])
        halo_percent[i,0] = a
        halo_percent[i,1] = b
        halo_percent[i,2] = c
        halo_percent[i,3] = e
        halo_percent[i,4] = d
        a, b, c, d, e = np.percentile(frac_Msham_hist[rm_NaN,i], [2.5, 16, 50, 84, 97.5])
        sham_percent[i,0] = a
        sham_percent[i,1] = b
        sham_percent[i,2] = c
        sham_percent[i,3] = e
        sham_percent[i,4] = d

    halo_sub.fill_between(t_cosmic, halo_percent[:,0], halo_percent[:,-1], 
            color=pretty_colors[1], alpha=0.25, edgecolor=None, lw=0) 
    halo_sub.fill_between(t_cosmic, halo_percent[:,1], halo_percent[:,-2], 
            color=pretty_colors[1], alpha=0.25, edgecolor=None, lw=0) 

    sham_sub.fill_between(t_cosmic, sham_percent[:,0], sham_percent[:,-1], 
            color=pretty_colors[3], alpha=0.25, edgecolor=None, lw=0) 
    sham_sub.fill_between(t_cosmic, sham_percent[:,1], sham_percent[:,-2], 
            color=pretty_colors[3], alpha=0.25, edgecolor=None, lw=0) 
    
    halo_sub.set_yscale("log")
    halo_sub.set_xlim([5, 13.5]) 
    halo_sub.set_ylim([10**-2, 10.]) 
    sham_sub.set_yscale("log")
    sham_sub.set_xlim([5, 13.5]) 
    sham_sub.set_ylim([10**-2, 10.]) 
    plt.show() 
    plt.close()


def SHAM_History(): 
    ''' Investigate the accretion histories from SHAM 
    '''
    # Read CenQue object 
    galpop = CMS.Read_CenQue('sfms', cenque='default')

    Mhalo_hist = galpop.Mhalo_hist
    Msham_hist = galpop.Msham_hist
    
    z_acc, t_acc = UT.zt_table()
    t_cosmic  = t_acc[1:16]
    
    m_bin = np.where((Mhalo_hist[:,-1] > 12.) & (Mhalo_hist[:,-1] < 12.5)) 
    #m_bin = np.where((galpop.halo_mass > 12.5) & (galpop.halo_mass < 13.)) 

    Mhalo_hist_bin = Mhalo_hist[m_bin, :][0]
    Msham_hist_bin = Msham_hist[m_bin, :][0]
    
    prettyplot() 
    pretty_colors = prettycolors() 
    fig = plt.figure()
    halo_sub = fig.add_subplot(121)
    sham_sub = fig.add_subplot(122)
    
    halo_percent = np.zeros((len(t_cosmic), 5))
    sham_percent = np.zeros((len(t_cosmic), 5))
    for i in xrange(Mhalo_hist.shape[1]): 
        rm_NaN = np.where(Mhalo_hist_bin[:,i] != -999.) 
        a, b, c, d, e = np.percentile(10**Mhalo_hist_bin[rm_NaN,i], [2.5, 16, 50, 84, 97.5])
        halo_percent[i,0] = a
        halo_percent[i,1] = b
        halo_percent[i,2] = c
        halo_percent[i,3] = d
        halo_percent[i,4] = e
        a, b, c, d, e = np.percentile(10**Msham_hist_bin[rm_NaN,i], [2.5, 16, 50, 84, 97.5])
        sham_percent[i,0] = a
        sham_percent[i,1] = b
        sham_percent[i,2] = c
        sham_percent[i,3] = d
        sham_percent[i,4] = e

    halo_sub.fill_between(t_cosmic, halo_percent[:,0], halo_percent[:,-1], 
            color=pretty_colors[1], alpha=0.25, edgecolor=None, lw=0) 
    halo_sub.fill_between(t_cosmic, halo_percent[:,1], halo_percent[:,-2], 
            color=pretty_colors[1], alpha=0.5, edgecolor=None, lw=0) 

    sham_sub.fill_between(t_cosmic, sham_percent[:,0], sham_percent[:,-1], 
            color=pretty_colors[3], alpha=0.25, edgecolor=None, lw=0) 
    sham_sub.fill_between(t_cosmic, sham_percent[:,1], sham_percent[:,-2], 
            color=pretty_colors[3], alpha=0.5, edgecolor=None, lw=0) 
    
    halo_sub.set_yscale("log")
    halo_sub.set_xlim([5, 13.5]) 
    halo_sub.set_ylim([10**11.5, 10**13]) 
    sham_sub.set_yscale("log")
    sham_sub.set_xlim([5, 13.5]) 
    sham_sub.set_ylim([10**9.5, 10**11.5]) 
    plt.show() 
    plt.close()


if __name__=='__main__': 
    SHAM_History()
