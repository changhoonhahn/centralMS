'''

Create catalog for centralMS project. 


'''
import os
import h5py
import numpy as np
from astropy.io import fits 
# --- local --- 
from . import util as UT
from . import sham_hack as sham

# modules only available on Harmattan or Sirocco
try: 
    from treepm import subhalo_io 
    from utilities import utility as wetzel_util
except ImportError:
    pass


class Subhalos(object): 
    def __init__(self, sigma_smhm=0.2, smf_source='li-march', nsnap0=20): 
        ''' catalog of subhalos over the snapshots 1 to nsnap0
        
        Parameters
        ----------
        * sigma_smhm : float
            Scatter in the Stellar Mass to Halo Mass relation. Default is 0.2
        * smf_source : string 
            SMF choice for SHAM. Default is Li et al.(2009) at z~0 interpolated at
            higher z with Marchesini et al.(2009)
        * nsnap_ancestor : int 
            Snapshot limit to track back to. 
        '''
        self.sigma_smhm = sigma_smhm # stellar mass to halo mass relation 
        self.smf_source = smf_source
        self.nsnap0 = nsnap0 

    def File(self): 
        '''
        '''
        file_name = ''.join([UT.dat_dir(), 
            'Subhalos', 
            '.SHAMsig', str(self.sigma_smhm), 
            '.smf_', self.smf_source, 
            '.nsnap0_', str(self.nsnap0), 
            '.hdf5']) 
        return file_name 

    def Read(self, **kwargs): 
        ''' Read in the hdf5 file specified in self.File()
        to a simple dictionary
        '''
        f = h5py.File(self.File(**kwargs), 'r') 
        catalog = {} 
        catalog['metadata'] = {}
        for key in f.attrs.keys(): 
            catalog['metadata'][key] = f.attrs[key]
        for key in f.keys():
            catalog[key] = f[key].value
        return catalog 

    def _Build(self, silent=True): 
        '''Construct the subhalo/SHAM M* history into hdf5 format
        '''
        # import subhalos from TreePM from snapshots 1 to nsnap0+10 
        # the earlier snapshots are included to have a more complete 
        # halo history
        sub = subhalo_io.Treepm.read('subhalo', 250, zis=range(1, self.nsnap0+10)) 

        # SHAM stellar masses to each of the subhalos at 
        # snapshots 1 to nsnap0
        sham.assign(sub, 'm.star', scat=self.sigma_smhm, 
                dis_mf=0.0, source=self.smf_source, zis=range(1,self.nsnap0+1)) 
        
        n_halo = sub[1]['halo.m'].shape[0]
        # save snapshot 1 properties 
        catalog = {} 
        for prop in ['halo.m', 'm.max', 'm.star', 'ilk']: 
            if prop == 'm.star': 
                catalog['m.sham'] = sub[1][prop]
            else: 
                catalog[prop] = sub[1][prop]
        
        # central/satellite classification
        central_indices = wetzel_util.utility_catalog.indices_ilk(sub[1], ilk='cen') 
        catalog['central'] = np.zeros(n_halo).astype('int')
        catalog['central'][central_indices] = 1

        for i_snap in range(2,self.nsnap0+10): # 2 to nsnap0+10
            # find subhalos that correspond to snapshot 1 subhalos 
            index_history = wetzel_util.utility_catalog.indices_tree(sub, 1, i_snap, range(n_halo))
            assert len(index_history) == n_halo 
            
            hasmatch = (index_history >= 0)
            index_history[~hasmatch] = -999  # subhalos w/ no match 
            catalog['index.snap'+str(i_snap)] = index_history
        
            if not silent: 
                print('%f of the subhalos have match at snapshot %i' %
                        (np.float(np.sum(hasmatch))/np.float(n_halo), i_snap))
            
            # save specified subhalo properties from the snapshots
            if i_snap <= self.nsnap0: props = ['halo.m', 'm.max', 'm.star', 'ilk']
            else: props = ['halo.m', 'm.max']

            for prop in props: 
                empty = np.repeat(-999., n_halo)
                if prop == 'ilk': empty = empty.astype(int) 
                if prop == 'm.star':
                    catalog['m.sham.snap'+str(i_snap)] = empty  
                    catalog['m.sham.snap'+str(i_snap)][hasmatch] = (sub[i_snap][prop])[index_history[hasmatch]]
                else: 
                    catalog[prop+'.snap'+str(i_snap)] = empty  
                    catalog[prop+'.snap'+str(i_snap)][hasmatch] = (sub[i_snap][prop])[index_history[hasmatch]]
            
            if i_snap <= self.nsnap0: 
                # classify central/satellite at snapshot i_snap
                central_indices = wetzel_util.utility_catalog.indices_ilk(sub[i_snap], ilk='cen') 
                central_snap = np.zeros(len(sub[i_snap]['halo.m'])).astype(int)
                central_snap[central_indices] = 1  # 1 = central 0 = not central
                catalog['central.snap'+str(i_snap)] = np.zeros(n_halo).astype(int)
                catalog['central.snap'+str(i_snap)][hasmatch] = central_snap[index_history[hasmatch]]

        if not silent: 
            # check that every subhalos have some history
            hasmatches = np.ones(n_halo).astype(bool) 
            
            for i_snap in range(2, self.nsnap0+1):  
                hasmatches = (hasmatches & (catalog['index.snap'+str(i_snap)] > 0)) 
            print('%f of the subhalos at nsnap=1 have parent subhalos out to %i' % 
                    (np.sum(hasmatches)/np.float(len(hasmatches)), self.nsnap0))
            mstarlim = (catalog['m.sham'] > 9.0)
            print('%f of the subhalos with M*_sham > 9. at nsnap=1 have parent subhalos out to %i' % 
                    (np.sum(hasmatches & mstarlim)/np.float(np.sum(mstarlim)), self.nsnap0)) 

        # go through snapshots and get when m.star > 0 
        nsnap_start = np.repeat(-999, n_halo) 
        for i_snap in range(1,self.nsnap0+1)[::-1]: 
            if i_snap > 1: 
                mstar = catalog['m.sham.snap'+str(i_snap)]
            else: 
                mstar = catalog['m.sham']
            nsnap_start[(mstar > 0.) & (nsnap_start < i_snap)] = i_snap 
        catalog['nsnap_start'] = nsnap_start
    
        # get m.sham at the initial snapshots of the halo 
        catalog['m.star0'] = np.zeros(ngal) # initial SHAM stellar mass 
        catalog['halo.m0'] = np.zeros(ngal) # initial halo mass 
        for i in range(1, nsnap0+1): 
            istart = (catalog['nsnap_start'] == i) # subhalos that being at snapshot i  
            str_snap = ''
            if i != 1: str_snap = '.snap'+str(i) 
            catalog['m.star0'][istart] = catalog['m.sham'+str_snap][istart]
            catalog['halo.m0'][istart] = catalog['halo.m'+str_snap][istart]

        # write to hdf5 file witht he simplest organization
        f = h5py.File(self.File(), 'w') 
        # -- meta data -- 
        f.attrs['nsnap0'] = self.nsnap0
        f.attrs['sigma_smhm'] = self.sigma_smhm
        f.attrs['smf_source'] = self.smf_source
        # -- data -- 
        for key in catalog.keys(): 
            f.create_dataset(key, data=catalog[key])
        f.close() 
        return None 


class CentralSubhalos(Subhalos): 
    def __init__(self, sigma_smhm=0.2, smf_source='li-march', nsnap0=15): 
        ''' Central Subhalo that have been centrals throughout
        their history. This is to remove splashback (ejected 
        satellite) galaxies.
        
        Parameters
        ----------
        * sigma_smhm : float
            Scatter in the Stellar Mass to Halo Mass relation. Default is 0.2
        * smf_source : string 
            SMF choice for SHAM. Default is Li et al.(2009) at z~0 interpolated at
            higher z with Marchesini et al.(2009)
        * nsnap_ancestor : int 
            Snapshot limit to track back to. 
        '''
        self.sigma_smhm = sigma_smhm # stellar mass to halo mass relation 
        self.smf_source = smf_source
        self.nsnap0 = nsnap0
    
    def File(self, downsampled=False): 
        ''' File name of the hdf5 file  
        '''
        str_down = ''
        if downsampled: 
            str_down = '.down'+downsampled+'x'

        file_name = ''.join([UT.dat_dir(), 
            'CentralSubhalos', 
            '.SHAMsig', str(self.sigma_smhm), 
            '.smf_', self.smf_source, 
            '.nsnap0_', str(self.nsnap0), 
            str_down, '.hdf5']) 
        return file_name 

    def _Build(self, silent=True): 
        '''Construct the pure central subhalo/SHAM M* history into hdf5 format
        '''
        # read in the entire subhalo history
        subhist = Subhalos(
                sigma_smhm=self.sigma_smhm, # scatter in SMHMR 
                smf_source=self.smf_source, # SMF from the literature
                nsnap0=self.nsnap0)         # initial snapshot 
        shcat = subhist.Read()

        iscen = (shcat['central'] == 1)     # subhalos centrals at nsnap=1 
        n_central = np.sum(iscen)
        for i_snap in range(2,self.nsnap0+1):    # 2 to nsnap_ancestor
            iscen = (iscen & (shcat['central.snap'+str(i_snap)] == 1))
    
        if not silent: 
            print('%i of %i subhalos are centrals at nsnap=1' % 
                    (n_central, len(shcat['central'])))
            print('%i of %i central subhalos at nsnap=1 are centrals throughout nsnap=1 to %i' %
                    (np.sum(iscen), n_central, self.nsnap0)) 
            mslim = (shcat['m.sham'] > 9.) 
            print('%f central subhalos w/ M* > 9. at nsnap=1 are throughout nsnap=1 to %i' %
                    (np.float(np.sum(iscen & mslim))/float(np.sum((shcat['central'] == 1) & mslim)), self.nsnap0))
        
        # make sure mhalo and mmax are both non-zero 
        # impose a halo.m cut, which was determined so that the 
        # sample is stellar masses complete out to M_*~10^9. Below 
        # things get gnarly. 
        cut_mass = ((shcat['m.max'] > 0.) & (shcat['halo.m'] > 10.6))
        cut_tot = (iscen & cut_mass) 
        if not silent: 
            print('%i of %i subhalos that are centrals throughout are above the mass limit' % 
                    (np.sum(cut_tot), np.sum(iscen)))
    
        catalog = {} 
        for key in shcat.keys(): 
            if key == 'metadata': continue 
            catalog[key] = shcat[key][cut_tot]

        # save to hdf5 file 
        if not silent: print('writing to %s ... ' % self.File()) 
        f = h5py.File(self.File(), 'w') 
        # -- meta data -- 
        f.attrs['nsnap0'] = self.nsnap0
        f.attrs['sigma_smhm'] = self.sigma_smhm
        f.attrs['smf_source'] = self.smf_source
        # -- data -- 
        for key in catalog.keys(): 
            f.create_dataset(key, data=catalog[key])
        f.create_dataset('weights', data=np.repeat(1., np.sum(cut_tot)))
        f.close() 
        return None 

    def Downsample(self, nbin_thresh=4000, dmhalo=0.2, silent=True): 
        ''' Downsample the catalog within bins of halo mass 
        '''
        catalog = self.Read()
        nhalo = len(catalog['halo.m']) 
        mhalo_min = catalog['halo.m'].min()
        mhalo_max = catalog['halo.m'].max() 
        weights = catalog['weights'] 
        if not silent: print('%i subhalo with %f < Mhalo < %f' % (nhalo, mhalo_min, mhalo_max))

        # halo mass bins
        mhbin = np.arange(0.2*np.floor(mhalo_min/0.2), 0.2*(np.ceil(mhalo_max/0.2)+1), 0.2) 
        
        for i_m in range(len(mhbin)-1): 
            in_mhbin = ((catalog['halo.m'] > mhbin[i_m]) & (catalog['halo.m'] <= mhbin[i_m+1]))
            nbin = np.sum(in_mhbin)

            if nbin > nbin_thresh:  
                weights[in_mhbin] = 0.
                f_down = float(nbin_thresh)/float(nbin) 

                keep_ind = np.random.choice(range(nbin), nbin_thresh, replace=False) 
                weights[np.arange(nhalo)[in_mhbin][keep_ind]] = 1./f_down
    
        f_down = (float(nhalo)/float(np.sum(weights > 0.)))
        if not silent: print('downsampled by %f x' % f_down)
        
        assert np.allclose(float(nhalo), np.sum(weights))

        hasw = np.where(weights > 0.)
        
        # ouptut downsampled catalog to hdf5 file  
        down_file = self.File(downsampled=str(int(round(f_down))))
        if not silent: print('writing to %s ... ' % down_file)
        f = h5py.File(down_file, 'w') 
        # -- meta data -- 
        f.attrs['nsnap0'] = self.nsnap0
        f.attrs['sigma_smhm'] = self.sigma_smhm
        f.attrs['smf_source'] = self.smf_source
        f.attrs['nbin_thresh'] = nbin_thresh
        f.attrs['dmhalo'] = dmhalo
        # -- data -- 
        for key in catalog.keys(): 
            if key in ['weights', 'metadata']: continue 
            f.create_dataset(key, data=catalog[key][hasw]) 
        f.create_dataset('weights', data=weights[hasw])
        f.close() 
        return None 


class SDSSgroupcat(Subhalos): 
    ''' Subhalos subclass to deal with SDSS group catalog data 
    '''
    def __init__(self, Mrcut=18, censat='central'):
        '''
        '''
        mrmasscut_dict = {18: '9.4', 19: '9.8', 20: '10.2'}
        self.Mrcut = Mrcut
        self.masscut = mrmasscut_dict[Mrcut]
        self.censat = censat 
        self.h = 0.7
        if self.censat not in ['central', 'satellite', 'all']: 
            raise ValueError

    def File(self): 
        ''' file name 
        '''
        f = ''.join([UT.dat_dir(), 'observations/',
            'SDSSgroupcat',
            '.Mr', str(self.Mrcut), 
            '.Mstar', str(self.masscut), 
            '.', str(self.censat), 
            '.hdf5']) 
        return f 
    
    def _Build(self, silent=True):  
        '''
        '''
        # load in galdata .fits file  
        fgal = fits.open(''.join([UT.dat_dir(), 'observations/', 
            'clf_groups_M', str(self.Mrcut), '_', str(self.masscut), '_D360.', 
            'galdata_corr.fits']) )
        gal_data = fgal[1].data 
    
        catalog = {} 
        catalog['ra']   = gal_data['ra'] * 57.2957795
        catalog['dec']  = gal_data['dec'] * 57.2957795
        catalog['z']    = gal_data['cz']/299792.458
        catalog['mstar'] = np.log10(gal_data['stellmass'] / self.h**2)   # convert to log Mass
        catalog['ssfr'] = gal_data['ssfr'] + np.log10(self.h**2) # remove h dependence 
        catalog['sfr']  = catalog['ssfr'] + catalog['mstar']

        # read in central/satellite probability file 
        fprob = fits.open(''.join([UT.dat_dir(), 'observations/', 
            'clf_groups_M', str(self.Mrcut), '_', str(self.masscut), '_D360.', 'prob.fits'])) 
        prob_data = fprob[1].data
        catalog['p_sat'] = prob_data['p_sat'] 

        # central or satellite 
        if self.censat == 'central': 
            censat_cut = (prob_data['p_sat'] <= 0.5) 
        elif self.censat == 'satellite': 
            censat_cut = (prob_data['p_sat'] > 0.5) 
        elif self.censat == 'all': 
            censat_cut = np.ones(len(prob_data['p_sat'])).astype(bool) 
        if not silent: 
            print('%i out of %i galaxies are %s' % (np.sum(censat_cut), len(prob_data['p_sat']), self.censat))
        
        # write to hdf5 file 
        if not silent: print('writing to %s ... ' % self.File()) 
        f = h5py.File(self.File(), 'w') 
        for key in catalog.keys(): 
            f.create_dataset(key, data=catalog[key][censat_cut])
        f.close() 
        return None 
