import os
import numpy as np 
# -- centralms -- 
from centralms import util as UT
from centralms import sfh as SFH 
from centralms import abcee as ABC
from centralms import evolver as Evol
# -- matplotlib -- 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from ChangTools.plotting import prettycolors
from matplotlib.patches import Rectangle
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
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['hatch.linewidth'] = 0.3  


def model_sfh(nsnap0=15, downsampled='20'):
    ''' SFH of galaxy  
    '''
    # get median of ABC posterior
    method = 'randomSFH0.5gyr.sfsmf.sfsbroken'
    abcout = ABC.readABC(method, 14)
    theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) for i in range(len(abcout['theta'][0]))]
        
    # run model on theta_median 
    tt = ABC._model_theta(method, theta_med)

    censub  = Cat.CentralSubhalos(nsnap0=nsnap0)
    shcat   = censub.Read(downsampled=downsampled) 

    # meta data 
    nsnap0 = shcat['metadata']['nsnap0']
    ngal = len(shcat['m.sham'])

    shcat = Evol.initSF(shcat, tt) # get SF halos  
    isSF = np.arange(ngal)[shcat['galtype'] == 'sf']

    # initiate logSFR(logM, z) function and keywords
    logSFR_logM_z, sfr_kwargs = SFH.logSFR_initiate(shcat, isSF, 
            theta_sfh=tt['sfh'], theta_sfms=tt['sfms'], testing=True)
    
    tage_i = UT.t_snap(nsnap0) 
    tage_f = UT.t_snap(0) 
    tage_arr = np.linspace(tage_i, tage_f, int((tage_f - tage_i)/0.1)) 

    # get integrated stellar masses 
    logM_integ, logSFRs = Evol._MassSFR_tarr(
            shcat, 
            tage_arr 
            isSF=isSF, 
            logSFR_logM_z=logSFR_logM_z, 
            sfr_kwargs=sfr_kwargs,
            theta_sfh=theta['sfh'], 
            theta_sfms=theta['sfms'], 
            theta_mass=theta['mass'])

    logSFR_t = np.empty(logM_integ.shape) 
    for i in range(logM_integ.shape[0]): 
        logSFR_t[i,:] = logsfr_M_z(logM_integ[i,:], zoft(tage_arr), **sfr_kwargs) 

    # keep galaxies with full SFHs and w > 0 
    keep = ((subcat['nsnap_start'][isSF] == nsnap0) & (subcat['weights'][isSF] > 0.)) 
    

    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)

    for i in np.random.choice(np.sum(keep), size=10): 
        sub.plot(tage_arr, logSFR_t[isSF,:][keep][i]) 

    sub.set_xlabel('$t$', fontsize=25)
    sub.set_xlim([5., 13.7])
    fig.savefig(''.join([UT.tex_dir(), 'figs/sfh_for_jeremy.png']), bbox_inches='tight' )
    return None 

