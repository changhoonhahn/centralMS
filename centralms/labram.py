'''

Use Louis's SFHs to look at star-forming galaxies


'''
import numpy as np 
import h5py 

import util as UT
import observables as Obv
import catalog as Cats
from ChangTools.fitstables import mrdfits


def read_LA(nsnap): 
    ''' return log(Mstar) and log(SFR) at snapshot = nsnap 
    '''
    f = ''.join([UT.dat_dir(), 'labramson/', 'mikeGenBasic.fits'])
    labram = mrdfits(f)

    iz = np.abs(labram.time[0] - UT.t_nsnap(nsnap)).argmin() 
    mstar = np.log10(labram.mstel_t[0,:,iz])
    sfr = np.log10(labram.sfr_t[0,:,iz]) - 9.0
    return [mstar, sfr]


def matchSMF(nsnap, logM_range=[10., 12.], dlogM=0.1, smf_source='li-march'): 
    ''' Return galaxies weights for Lousis's galaxies that would match
    the snapshot = nsnap SMF. 
    '''
    mstar, _ = read_LA(nsnap) # read in LA's galaxies at nsnap
    
    # calculate the weight in stellar mass bins
    mbin = np.arange(logM_range[0], logM_range[1]+dlogM, dlogM)
    ngal, bbb = np.histogram(mstar, bins=mbin)
    phi = Obv.analyticSMF(UT.z_nsnap(nsnap), m_arr=mbin[:-1], dlogm=mbin[1]-mbin[0], source=smf_source)
    w_mbin = np.array([phi[1][i]/float(ngal[i]) for i in range(len(ngal))])
    
    # assign stellar mass bin weights to each galaxy 
    ws = np.zeros(len(mstar))
    for im in range(len(mbin)-1): 
        inmbin = np.where((mstar >= mbin[im]) & (mstar < mbin[im+1]))
        if len(inmbin[0]) > 0: 
            ws[inmbin] = w_mbin[im]
    return ws


def assignHalo(nsnap, halos, method='random', logM_range=[9., 12.], dlogM=0.2, SHAM_sig=0.2, nsnap0=15): 
    ''' assign halos to LA's galaxies by binning the SHAM halo catalog
    in log M* bins and then assignign them halos. Need to test that this 
    preserves sigma_M* at snapshot nsnap
    '''
    # get M_halo and M_SHAM from SHAM-ed halo catalog
    if nsnap == 1: 
        str_snap = ''
    else: 
        str_snap = 'snapshot'+str(nsnap)+'_'
    mhalo = halos[str_snap+'m.max']
    msham = halos[str_snap+'m.star']
    mstar, _ = read_LA(nsnap) # read in LA's galaxies at nsnap

    if method == 'dmhalo': 
        # halos are assigned such that galaxies with the most stellar
        # mass growth are assigned to halos with the most halo mass growth
        # over z ~ 1 to 0 
        if nsnap == 1: 
            raise ValueError
        mstar_f, _ = read_LA(1)
        dmstar = mstar_f - mstar
        dmhalo = halos['m.max'] - mhalo 

    index_assign = np.repeat(-999, len(mstar))
    for mlow in np.arange(logM_range[0], logM_range[1], dlogM): 
        sham_mbin = np.where((msham >= mlow) & (msham < mlow+dlogM))[0]
        labm_mbin = np.where((mstar >= mlow) & (mstar < mlow+dlogM))[0]
        if len(sham_mbin) > 0 and len(labm_mbin) > 0: 
            if method == 'random': 
                index_assign[labm_mbin] = np.random.choice(sham_mbin, size=len(labm_mbin), replace=False)
            elif method == 'dmhalo': 
                dmstar_bin = dmstar[labm_mbin]
                i_dmstar_sort = np.argsort(dmstar_bin)

                i_rand = np.random.choice(sham_mbin, size=len(labm_mbin), replace=False)
                i_sort = np.argsort(dmhalo[i_rand])

                index_assign[labm_mbin[i_dmstar_sort]] = i_rand[i_sort]
            else:
                raise ValueError
    return index_assign
