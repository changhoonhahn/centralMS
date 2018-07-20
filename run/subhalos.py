'''
'''
import os
import h5py
import numpy as np
# -- centralms -- 
from centralms import util as UT
from centralms import sham_hack as sham
# -- treepm -- 
try: 
    from treepm import subhalo_io 
    from utilities import utility as wetzel_util
except ImportError:
    pass


def buildSubhalos(nsnap0=20, sigma_smhm=0.2, smf_source='li-march'): 
    '''
    '''
    # import subhalos from TreePM from snapshots 1 to nsnap0+10 
    # the earlier snapshots are included to have a more complete 
    # halo history
    sub = subhalo_io.Treepm.read('subhalo', 250, zis=range(1, nsnap0+10)) 

    # SHAM stellar masses to each of the subhalos at 
    # snapshots 1 to nsnap0
    sham.assign(sub, 'm.star', scat=self.sigma_smhm, 
            dis_mf=0.0, source=self.smf_source, zis=self.snapshots) 
    
    n_halo = sub[1].shape[0]
    # save snapshot 1 properties 
    catalog = {} 
    for prop in ['halo.m', 'm.max', 'm.star', 'ilk']: 
        catalog[prop] = sub[1][prop]
    
    # central/satellite classification
    central_indices = wetzel_util.utility_catalog.indices_ilk(sub[1], ilk='cen') 
    catalog['central'] = np.zeros(n_halo).astype('int')
    catalog['central'][central_indices] = 1

    for i_snap in range(2,nsnap0+10): # 2 to nsnap0+10
        # find subhalos that correspond to snapshot 1 subhalos 
        index_history = wetzel_util.utility_catalog.indices_tree(sub, 1, i_snap, n_halo)
        assert len(index_history) == n_halo 
        
        hasmatch = (index_history >= 0)
        index_history[~hasmatch] = -999  # subhalos w/ no match 
        catalog['index.snap'+str(i_snap)] = index_history

        print('%f of the subhalos have match at snapshot %i' %
                (np.float(np.sum(hasmatch))/np.float(n_halo), i_snap))
        
        # save specified subhalo properties from the snapshots
        if i_snap <= nsnap0: props = ['halo.m', 'm.max', 'm.star', 'ilk']
        else: props = ['halo.m', 'm.max']

        for prop in props: 
            empty = np.repeat(-999., n_halo)
            if prop == 'ilk': empty = empty.astype(int) 
            catalog[prop+'.snap'+str(i_snap)] = empty  
            catalog[prop+'.snap'+str(i_snap)][hasmatch] = (sub[i_snap][prop])[index_history[hasmatch]]
        
        if i_snap <= nsnap0: 
            # classify central/satellite at snapshot i_snap
            central_indices = wetzel_util.utility_catalog.indices_ilk(sub[i_snap], ilk='cen') 
            central_snap = np.zeros(n_halo).astype(int)
            central_snap[central_indices] = 1  # 1 = central 0 = not central
            catalog['central.snap'+str(i_snap)] = np.zeros(n_halo).astype(int)
            catalog['central.snap'+str(i_snap)][hasmatch] = central_snap[index_history[hasmatch]]
        

    # go through snapshots and get when m.star > 0 
    nsnap_start = np.repeat(-999, n_halo) 
    for i_snap in range(1,nsnap0+1)[::-1]: 
        if i_snap > 1: 
            mstar = catalog['m.star.snap'+str(i_snap)]
        else: 
            mstar = catalog['m.star']
        nsnap_start[(mstar > 0.) & (nsnap_start < i_snap)] = i_snap 
    catalog['nsnap_start'] = nsnap_start

    # save catalog to hdf5
    f_sh = ''.join([UT.dat_dir(), 'Subhalos', 
        '.SHAMsig', str(sigma_smhm),    # scatter
        '.smf_', self.smf_source,       # SMF 
        '.nsnap0_', str(self.nsnap0),   # initial snapshot 
        '.hdf5']) 

    f = h5py.File(f_sh, 'w') 
    for key in catalog.keys(): 
        f.create_dataset(key, data=catalog[key])
    f.close() 
    return None 


if __name__=="__main__":
    buildSubhalos(nsnap0=20, sigma_smhm=0.2, smf_source='li-march')
