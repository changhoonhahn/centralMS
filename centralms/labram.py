'''

Use Louis's SFHs to look at star-forming galaxies

'''
import numpy as np 
import scipy as sp 
from astropy.io import fits 
# --- centralMS ---
from centralms import util as UT
from centralms import observables as Obvs


def catalogAbramson2016(): 
    ''' Read in the SFHs used in Abramson et al.(2016), generated
    by Gladders.
    '''
    # read in the file
    f_la2016 = fits.open(''.join([UT.dat_dir(), 'labramson/mikeGenBasic.fits']))
    la2016 = f_la2016[1].data 

    # Columns: ['TIME', 'REDSHIFT', 'TODAY', 'MSTEL_T', 'MSTEL_OBS', 'SFR_T', 'SFR_OBS', 'SFR_MOD_SNAP', 'MSTEL_FINAL', 'SNAP', 'T_TO_M_OBS', 'T_TO_M_FINAL', 'T0', 'TAU']
    cat = {} 
    for name in la2016.names: 
        if name != 'SFR_T': 
            cat[name] = la2016[name][0]
        else: 
            cat[name] = la2016[name][0] / 1e9
    return cat


def SHassignSFH(sh, nsnap=15, logMs_range=[9.,12.]): 
    ''' Assign Louis's SFHs to the subhalo catalog based on their Msham. 
    '''
    # read in Louis's catalog
    cat = catalogAbramson2016() 
    i_z = np.argmin(np.abs(cat['REDSHIFT'] - UT.z_nsnap(nsnap))) #  index that best match to snapshot 
    mstar = np.log10(cat['MSTEL_T'][:,i_z]) # M* @ z ~ zsnap
    
    str_snap = ''
    if nsnap != 1: 
        str_snap = '.snap'+str(nsnap)

    msham = sh['m.sham'+str_snap]
    mlim = np.where((msham > logMs_range[0]) & (msham < logMs_range[1]))[0]
    match_la2016 = np.repeat(-999, len(msham)) 
    for i_m in mlim: 
        match_la2016[i_m] = np.argmin(np.abs(msham[i_m] - mstar))
    return match_la2016


def matchSMF(logms_z, z, logM_range=[10., 12.], dlogM=0.1, smf_source='li-march'): 
    ''' Return galaxies weights for Lousis's galaxies so that his galaxies
    would match SMF at snapshot nsnap. 
    '''
    # `smf_source` analytic SMF at z=zsnap
    mbin = np.arange(logM_range[0], logM_range[1], dlogM)
    mbin = np.concatenate([mbin, [mbin[-1] + dlogM]]) 
    phi = Obvs.analyticSMF(z, m_arr=mbin[:-1], dlogm=mbin[1]-mbin[0], source=smf_source)

    # calculate the weight in stellar mass bins
    _, ngal = Obvs.getMF(logms_z, mbin=mbin)

    #w_int = sp.interpolate.interp1d(0.5*(mbin[1:]+mbin[:-1]), phi[1] / ngal.astype(float), kind='linear', fill_value='extrapolate')
    #ws = np.zeros(len(logms_z))
    #mlim = ((logms_z > logM_range[0]) & (logms_z < logM_range[1]))
    #ws[mlim] = w_int(logms_z[mlim])
    
    w_mbin = phi[1] / ngal.astype(float)
    # assign stellar mass bin weights to each galaxy 
    ws = np.zeros(len(logms_z))
    for im in range(len(mbin)-1): 
        inmbin = ((logms_z >= mbin[im]) & (logms_z < mbin[im+1]))
        if np.sum(inmbin) > 0: 
            ws[inmbin] = w_mbin[im]
    return ws


def assignHalo(mstar, nsnap, shcat, logM_range=[9., 12.], dlogMs=0.2): 
    ''' given mstar values for LA's galaxies, assign halos by binning the SHAM halo 
    catalog in log M* bins and then assignign them halos. 
    '''
    # read in Msham, Mmax, and Mhalo from subhalo catalog 
    str_snap = ''
    if nsnap != 1: 
        str_snap = '.snap'+str(nsnap)
    mhalo = shcat['halo.m'+str_snap]
    mmax = shcat['m.max'+str_snap]
    msham = shcat['m.sham'+str_snap]

    i_halo = np.repeat(-999, len(mstar))
    mlows = np.arange(logM_range[0], logM_range[1], dlogMs)
    for mlow in mlows: 
        sham_mbin = np.where((msham >= mlow) & (msham < mlow+dlogMs))[0]
        labm_mbin = np.where((mstar >= mlow) & (mstar < mlow+dlogMs))[0]
        if len(labm_mbin) > 0: 
            i_halo[labm_mbin] = np.random.choice(sham_mbin, size=len(labm_mbin), replace=False)
    return mhalo[i_halo], mmax[i_halo], i_halo
