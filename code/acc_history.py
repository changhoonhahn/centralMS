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


def SHAM_AccHistory(evol_dict=None): 
    ''' Investigate the accretion histories from SHAM 
    '''
    # Read CenQue object 
    galpop = CMS.EvolvedGalPop(cenque='default', evol_dict=evol_dict) 
    galpop.Read() 

    Mhalo_hist = galpop.Mhalo_hist
    Msham_hist = galpop.Msham_hist
    Minteg_hist = galpop.Minteg_hist[:, ::-1]
    
    z_acc, t_acc = UT.zt_table()
    t_cosmic  = t_acc[1:16]
    
    prettyplot() 
    pretty_colors = prettycolors() 
    fig = plt.figure(figsize=(15,15))
    bkgd = fig.add_subplot(111, frameon=False)
    
    mhalo_lims = [[11.5, 12.0], [12.0, 12.5], [12.5, 13.]]
    mstar_lims = [[10.25, 10.5], [10.5, 10.75], [10.75, 11.]]
    for i_m in range(len(mhalo_lims)):
        mhalo_lim = mhalo_lims[i_m]
        mstar_lim = mstar_lims[i_m]

        halo_sub = fig.add_subplot(3, 3, 1 + 3*i_m)
        sham_sub = fig.add_subplot(3, 3, 2 + 3*i_m)
        integ_sub = fig.add_subplot(3, 3, 3 + 3*i_m)

        mhalo_bin = np.where((galpop.halo_mass > mhalo_lim[0]) & (galpop.halo_mass < mhalo_lim[1])) 
        Mhalo_hist_bin = Mhalo_hist[mhalo_bin, :][0]
        frac_Mhalo_hist = 10.**(
                Mhalo_hist_bin - 
                np.tile(Mhalo_hist_bin[:,0], Mhalo_hist_bin.shape[1]).reshape(
                    Mhalo_hist_bin.shape[1], Mhalo_hist_bin.shape[0]).T
                )

        msham_bin = np.where((galpop.M_sham > mstar_lim[0]) & (galpop.M_sham < mstar_lim[1])) 
        Msham_hist_bin = Msham_hist[msham_bin, :][0]
        frac_Msham_hist = 10.**(
                Msham_hist_bin - 
                np.tile(Msham_hist_bin[:,0], Msham_hist_bin.shape[1]).reshape(
                    Msham_hist_bin.shape[1], Msham_hist_bin.shape[0]).T
                )

        minteg_bin = np.where((galpop.mass > mstar_lim[0]) & (galpop.mass < mstar_lim[1])) 
        Minteg_hist_bin = Minteg_hist[minteg_bin, :][0]
        frac_Minteg_hist = 10.**(
                Minteg_hist_bin - 
                np.tile(Minteg_hist_bin[:,0], Minteg_hist_bin.shape[1]).reshape(
                    Minteg_hist_bin.shape[1], Minteg_hist_bin.shape[0]).T
                )
        
        halo_percent = np.zeros((len(t_cosmic), 5))
        sham_percent = np.zeros((len(t_cosmic), 5))
        integ_percent = np.zeros((len(t_cosmic), 5))
        for i in xrange(Mhalo_hist.shape[1]): 
            rm_NaN_halo = np.where(Mhalo_hist_bin[:,i] != -999.) 
            halo_percent[i,:] = np.percentile(frac_Mhalo_hist[rm_NaN_halo,i],
                    [2.5, 16, 50, 84, 97.5])
            rm_NaN_sham = np.where(Msham_hist_bin[:,i] != -999.) 
            sham_percent[i,:] = np.percentile(frac_Msham_hist[rm_NaN_sham,i], 
                    [2.5, 16, 50, 84, 97.5])
            rm_NaN_integ = np.where(Minteg_hist_bin[:,i] != -999.) 
            integ_percent[i,:] = np.percentile(frac_Minteg_hist[rm_NaN_integ,i], 
                    [2.5, 16, 50, 84, 97.5])

        halo_sub.fill_between(t_cosmic, halo_percent[:,0], halo_percent[:,-1], 
                color=pretty_colors[1], alpha=0.25, edgecolor=None, lw=0) 
        halo_sub.fill_between(t_cosmic, halo_percent[:,1], halo_percent[:,-2], 
                color=pretty_colors[1], alpha=0.25, edgecolor=None, lw=0, 
                label=r"$"+str(round(mhalo_lim[0],1))+" < \mathtt{M^f_{halo}} < "+str(round(mhalo_lim[1],1))+"$") 
        for i in xrange(20): 
            if Mhalo_hist[i,:].min() > 0.:
                halo_sub.plot(t_cosmic, frac_Mhalo_hist[i,:], c=pretty_colors[1]) 

        sham_sub.fill_between(t_cosmic, sham_percent[:,0], sham_percent[:,-1], 
                color=pretty_colors[3], alpha=0.25, edgecolor=None, lw=0) 
        sham_sub.fill_between(t_cosmic, sham_percent[:,1], sham_percent[:,-2], 
                color=pretty_colors[3], alpha=0.25, edgecolor=None, lw=0, 
                label=r"$"+str(round(mstar_lim[0],1))+" < \mathtt{M^f_*} < "+str(round(mstar_lim[1],1))+"$") 
        for i in xrange(50): 
            if Msham_hist[i,:].min() > 0.:
                sham_sub.plot(t_cosmic, frac_Msham_hist[i,:], c=pretty_colors[3]) 
        
        integ_sub.fill_between(t_cosmic, integ_percent[:,0], integ_percent[:,-1], 
                color=pretty_colors[5], alpha=0.25, edgecolor=None, lw=0) 
        integ_sub.fill_between(t_cosmic, integ_percent[:,1], integ_percent[:,-2], 
                color=pretty_colors[5], alpha=0.25, edgecolor=None, lw=0,
                label=r"$"+str(round(mstar_lim[0],1))+" < \mathtt{M^f_*} < "+str(round(mstar_lim[1],1))+"$") 
        for i in xrange(50): 
            if Minteg_hist[i,:].min() != -999:
                integ_sub.plot(t_cosmic, frac_Minteg_hist[i,:], c=pretty_colors[5]) 
        
        halo_sub.set_yscale("log")
        halo_sub.set_xlim([5, 13.5]) 
        halo_sub.set_ylim([10**-2, 5.]) 
        halo_sub.legend(loc='upper left') 
        sham_sub.set_yscale("log")
        sham_sub.set_xlim([5, 13.5]) 
        sham_sub.set_ylim([10**-2, 5.]) 
        sham_sub.legend(loc='upper left') 
        integ_sub.set_yscale("log")
        integ_sub.set_xlim([5, 13.5]) 
        integ_sub.set_ylim([10**-2, 5.]) 
        integ_sub.legend(loc='upper left') 

    bkgd.set_xlabel(r'$\mathtt{t_{cosmic}}$ [Gyr]', fontsize=25)
    bkgd.set_ylabel(r'$\mathtt{M(t)/M_f}$', fontsize=25)

    fig_file = ''.join([UT.fig_dir(), 
        'AccHistory.MS', 
        '.sfr_', evol_dict['sfr']['name'], 
        '.mass_', evol_dict['mass']['type'], 
        '_', str(evol_dict['mass']['t_step']), 
        '.png']) 
    fig.savefig(fig_file, bbox_inches='tight') 
    plt.close()
    return None


def SHAM_History(evol_dict=None): 
    ''' Investigate the accretion histories from SHAM 
    '''
    # Read CenQue object 
    galpop = CMS.EvolvedGalPop(cenque='default', evol_dict=evol_dict) 
    galpop.Read() 

    Mhalo_hist = galpop.Mhalo_hist
    Msham_hist = galpop.Msham_hist
    Minteg_hist = galpop.Minteg_hist
    
    z_acc, t_acc = UT.zt_table()
    t_cosmic  = t_acc[1:16]
    
    m_bin = np.where((Msham_hist[:,-1] > 10.) & (Msham_hist[:,-1] < 10.5)) 
    #m_bin = np.where((Mhalo_hist[:,-1] > 12.) & (Mhalo_hist[:,-1] < 12.5)) 
    #m_bin = np.where((galpop.halo_mass > 12.5) & (galpop.halo_mass < 13.)) 

    Mhalo_hist_bin = Mhalo_hist[m_bin, :][0]
    Msham_hist_bin = Msham_hist[m_bin, :][0]
    Minteg_hist_bin = Minteg_hist[m_bin, :][0]
    
    prettyplot() 
    pretty_colors = prettycolors() 
    fig = plt.figure(figsize=(15,5))
    halo_sub = fig.add_subplot(131)
    sham_sub = fig.add_subplot(132)
    integ_sub = fig.add_subplot(133)
    
    halo_percent = np.zeros((len(t_cosmic), 5))
    sham_percent = np.zeros((len(t_cosmic), 5))
    integ_percent = np.zeros((len(t_cosmic), 5))
    for i in xrange(Mhalo_hist.shape[1]): 
        rm_NaN = np.where(Mhalo_hist_bin[:,i] != -999.) 
        halo_percent[i,:] = np.percentile(10**Mhalo_hist_bin[rm_NaN,i], [2.5, 16, 50, 84, 97.5])
        sham_percent[i,:] = np.percentile(10**Msham_hist_bin[rm_NaN,i], [2.5, 16, 50, 84, 97.5])
        integ_percent[i,:] = np.percentile(10**Minteg_hist_bin[rm_NaN,i], [2.5, 16, 50, 84, 97.5])
    
    halo_sub.fill_between(t_cosmic, halo_percent[:,0], halo_percent[:,-1], 
            color=pretty_colors[1], alpha=0.25, edgecolor=None, lw=0) 
    halo_sub.fill_between(t_cosmic, halo_percent[:,1], halo_percent[:,-2], 
            color=pretty_colors[1], alpha=0.5, edgecolor=None, lw=0) 

    sham_sub.fill_between(t_cosmic, sham_percent[:,0], sham_percent[:,-1], 
            color=pretty_colors[3], alpha=0.25, edgecolor=None, lw=0) 
    sham_sub.fill_between(t_cosmic, sham_percent[:,1], sham_percent[:,-2], 
            color=pretty_colors[3], alpha=0.5, edgecolor=None, lw=0) 
    
    integ_sub.fill_between(t_cosmic, integ_percent[:,0][::-1], integ_percent[:,-1][::-1], 
            color=pretty_colors[5], alpha=0.25, edgecolor=None, lw=0) 
    integ_sub.fill_between(t_cosmic, integ_percent[:,1][::-1], integ_percent[:,-2][::-1], 
            color=pretty_colors[5], alpha=0.5, edgecolor=None, lw=0) 
    
    halo_sub.set_yscale("log")
    halo_sub.set_xlim([5, 13.5]) 
    halo_sub.set_ylim([10**11.5, 10**13]) 
    sham_sub.set_yscale("log")
    sham_sub.set_xlim([5, 13.5]) 
    sham_sub.set_ylim([10**9.5, 10**11.5]) 
    integ_sub.set_yscale("log")
    integ_sub.set_xlim([5, 13.5]) 
    integ_sub.set_ylim([10**9.5, 10**11.5]) 
    plt.show() 
    plt.close()


if __name__=='__main__': 
    evol_dict = {
            'sfr': {'name': 'constant_offset'}, 
            'mass': {'type': 'rk4', 'f_retain': 0.6, 't_step': 0.01} 
            } 
    SHAM_AccHistory(evol_dict=evol_dict)
    evol_dict = {
            'sfr': {'name': 'no_scatter'}, 
            'mass': {'type': 'rk4', 'f_retain': 0.6, 't_step': 0.01} 
            } 
    SHAM_AccHistory(evol_dict=evol_dict)
