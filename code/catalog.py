'''

Create catalog for centralMS project. 


'''
import os
import h5py
import numpy as np

# --- Local --- 
import util as UT
from ChangTools.fitstables import mrdfits

# modules only available on Harmattan or Sirocco
import sham_hack as sham
try: 
    from treepm import subhalo_io 
    from utilities import utility as wetzel_util
except ImportError:
    pass


class SubhaloHistory(object): 
    def __init__(self, sigma_smhm=0.2, smf_source='li-march', nsnap_ancestor=20): 
        ''' Subhalo/SHAM M* history of SHAMed galaxies at snapshot 1. 
        
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
        self.nsnap_ancestor = nsnap_ancestor 
        self.snapshots = range(1, nsnap_ancestor+1)

    def File(self): 
        '''
        '''
        file_name = ''.join([UT.dat_dir(), 
            'SubhaloHistory', 
            '.sigma_SMHM', str(self.sigma_smhm), 
            '.smf_', self.smf_source, 
            '.Anc_nsnap', str(self.nsnap_ancestor), 
            '.hdf5']) 
        return file_name 

    def Read(self): 
        ''' Read in the hdf5 file 
        '''
        f = h5py.File(self.File(), 'r') 
        grp = f['data'] 
        catalog = {} 
        for key in grp.keys():
            catalog[key] = grp[key].value
        return catalog 

    def Build(self): 
        '''Construct the subhalo/SHAM M* history into hdf5 format
        '''
        # import that subhalos
        sub = subhalo_io.Treepm.read('subhalo', 250, zis=self.snapshots) 

        # SHAM stellar masses to each of the subhalo catalogs
        m_kind = 'm.star'
        sham.assign(sub, m_kind, scat=self.sigma_smhm, 
                dis_mf=0.0, source=self.smf_source, zis=self.snapshots) 

        catalog = {} 
        for i_snap in self.snapshots[1:]: # 2 to nsnap_ancestor
            # find subhalos that correspond to snapshot 1 subhalos 
            index_history = wetzel_util.utility_catalog.indices_tree(sub, 1, i_snap, range(len(sub[1]['halo.m'])))
            no_match = np.where(index_history < 0)[0]
            hasmatch = np.where(index_history > 0)[0]
            index_history[no_match] = -999  # ones that do not have a match  
            catalog['snapshot'+str(i_snap)+'_index'] = index_history
            
            if len(index_history) != len(sub[1]['halo.m']): 
                raise ValueError

            print np.float(len(hasmatch))/np.float(len(index_history)), ' of subhalos have matches'
            #print catalog['snapshot'+str(i_snap)+'_index'][:10]
            #print sub[1]['halo.m'][hasmatch][:10] 
            #print sub[i_snap]['halo.m'][catalog['snapshot'+str(i_snap)+'_index'][hasmatch]][:10]
            
            # save specified subhalo properties from the snapshots
            for gal_prop in ['halo.m', 'm.max', 'm.star', 'pos', 'ilk']: 
                if isinstance(sub[i_snap][gal_prop][0], np.float32) or isinstance(sub[i_snap][gal_prop][0], np.float64): 
                    catalog['snapshot'+str(i_snap)+'_'+gal_prop] = np.repeat(-999., len(catalog['snapshot'+str(i_snap)+'_index']))
                elif isinstance(sub[i_snap][gal_prop][0], np.int32): 
                    catalog['snapshot'+str(i_snap)+'_'+gal_prop] = np.repeat(-999, len(catalog['snapshot'+str(i_snap)+'_index']))
                elif isinstance(sub[i_snap][gal_prop][0], np.ndarray): 
                    catalog['snapshot'+str(i_snap)+'_'+gal_prop] = np.tile(-999, 
                            (len(catalog['snapshot'+str(i_snap)+'_index']), sub[i_snap][gal_prop].shape[1]))
                else: 
                    print gal_prop
                    print type(sub[i_snap][gal_prop][0])
                    raise ValueError
            
                catalog['snapshot'+str(i_snap)+'_'+gal_prop][hasmatch] = \
                        (sub[i_snap][gal_prop])[index_history[hasmatch]]

            # central indiex at snapshot i_snap
            central_indices = wetzel_util.utility_catalog.indices_ilk(sub[i_snap], ilk='cen') 
            central_snap = np.zeros(len(sub[i_snap]['halo.m']))
            central_snap[central_indices] = 1  # 1 = central 0 = not central

            central_history = np.repeat(0, len(sub[1]['halo.m']))
            central_history[hasmatch] = central_snap[index_history[hasmatch]]
            catalog['snapshot'+str(i_snap)+'_central'] = central_history 

        for gal_prop in ['halo.m', 'm.max', 'm.star', 'pos', 'ilk']: 
            catalog[gal_prop] = sub[1][gal_prop]

        # go through snapshots and get when m.star > 0 
        nsnap_start = np.repeat(-999, len(catalog['m.star'])) 
        for i_snap in self.snapshots[::-1]: 
            started = np.where((catalog['snapshot'+str(i_snap)+'_m.star'] > 0) & (nsnap_start < i_snap))
            nsnap_start[started] = i_snap 
        catalog['nsnap_start'] = nsnap_start

        central_indices = wetzel_util.utility_catalog.indices_ilk(sub[1], ilk='cen') 
        catalog['central'] = np.zeros(len(sub[1]['halo.m'])).astype('int')
        catalog['central'][central_indices] = 1

        # save catalog to hdf5
        file_name = self.File() 
        f = h5py.File(file_name, 'w') 
        grp = f.create_group('data') 
        for key in catalog.keys(): 
            grp.create_dataset(key, data=catalog[key])
        f.close() 
        return None 
    
    def _CheckHistory(self): 
        ''' Check the subhalo histories to make sure everything is going fine
        '''
        catalog = self.Read()
        
        hasmatches = np.repeat(True, len(catalog['halo.m'])) 
        for i_snap in self.snapshots[1:]: # 2 to nsnap_ancestor
            hasmatches = hasmatches & (catalog['snapshot'+str(i_snap)+'_index'] > 0)
        print np.sum(hasmatches)/np.float(len(hasmatches)), ' of the subhalos \nat nsnap = 1 has matches throughout' 
        print '---'

        stellmass_lim = (catalog['m.star'] > 9.0)
        print np.sum(hasmatches & stellmass_lim)/np.float(np.sum(stellmass_lim)), ' of the subhalos with M*_sham > 9. \nat nsnap = 1 has matches throughout' 

        return None


class PureCentralHistory(object): 
    def __init__(self, sigma_smhm=0.2, smf_source='li-march', nsnap_ancestor=20): 
        ''' Pure central Subhalo/SHAM M* history of SHAMed galaxies at snapshot 1. 
        
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
        self.nsnap_ancestor = nsnap_ancestor 
        self.snapshots = range(1, nsnap_ancestor+1)
    
    def File(self, downsampled=False): 
        ''' File name of the hdf5 file  
        '''
        str_down = ''
        if downsampled: 
            str_down = '.down'+downsampled+'x'
        file_name = ''.join([UT.dat_dir(), 
            'PureCentralHistory', 
            '.sigma_SMHM', str(self.sigma_smhm), 
            '.smf_', self.smf_source, 
            '.Anc_nsnap', str(self.nsnap_ancestor), 
            str_down,
            '.hdf5']) 
        return file_name 

    def Read(self, downsampled=False): 
        ''' Read in the hdf5 file 
        '''
        f = h5py.File(self.File(downsampled=downsampled), 'r') 
        grp = f['data'] 
        catalog = {} 
        for key in grp.keys():
            catalog[key] = grp[key].value
        return catalog 

    def Build(self): 
        '''Construct the pure central subhalo/SHAM M* history into hdf5 format
        '''
        # read in the entire subhalo history
        subhist = SubhaloHistory(sigma_smhm=self.sigma_smhm, smf_source=self.smf_source, nsnap_ancestor=self.nsnap_ancestor)
        all_catalog = subhist.Read()
        
        ispure = (all_catalog['central'] == 1)
        n_central = np.sum(ispure)
        print n_central, ' of', len(all_catalog['central']), ' subhalos at nsnap=1 are pure centrals'

        for i_snap in self.snapshots[1:]: # 2 to nsnap_ancestor
            ispure = ispure & (all_catalog['snapshot'+str(i_snap)+'_central'] == 1)

        print np.sum(ispure), ' of', n_central, ' central subhalos at nsnap=1 are pure (central throughout)'
        print np.float(np.sum(ispure & (all_catalog['m.star'] > 9.0)))/np.float(np.sum((all_catalog['central'] == 1) & (all_catalog['m.star'] > 9.0))), ' central subhalos w/ M* > 9. at nsnap=1 are pure throughout z < 1'
        print np.float(np.sum(ispure & (all_catalog['m.star'] > 9.5)))/np.float(np.sum((all_catalog['central'] == 1) & (all_catalog['m.star'] > 9.5))), ' central subhalos w/ M* > 9.5 at nsnap=1 are pure throughout z < 1'

        cut_mass = (all_catalog['m.star'] > 8.)
        cut_tot = np.where(ispure & cut_mass) 
    
        catalog = {} 
        for key in all_catalog.keys(): 
            catalog[key] = all_catalog[key][cut_tot]

        # save catalog
        file_name = self.File() 
        print 'writing to ... ', file_name 
        f = h5py.File(file_name, 'w') 
        grp = f.create_group('data') 
        for key in catalog.keys(): 
            grp.create_dataset(key, data=catalog[key])
        grp.create_dataset('weights', data=np.repeat(1., len(catalog['halo.m'])) )
        f.close() 

        return None 

    def Downsample(self, ngal_thresh=4000, dmhalo=0.2): 
        ''' Downsample the catalog 
        '''
        catalog = self.Read()
        Mhalo_min = np.min(catalog['halo.m'])
        Mhalo_max = np.max(catalog['halo.m'])
        print Mhalo_min, Mhalo_max

        mhalo_bin = np.arange(Mhalo_min - 0.5 * dmhalo, Mhalo_max + dmhalo, dmhalo) 
        
        weights = np.repeat(1., len(catalog['halo.m']))

        for i_m in range(len(mhalo_bin) - 1): 
            in_Mhalobin = np.where(
                    (mhalo_bin[i_m] < catalog['halo.m']) &
                    (mhalo_bin[i_m+1] >= catalog['halo.m'])) 
            ngal_inbin = len(in_Mhalobin[0])
            weights[in_Mhalobin] = 0.

            if ngal_inbin > ngal_thresh:  
                f_down = np.float(ngal_thresh)/np.float(ngal_inbin) 

                keep_ind = np.random.choice(range(ngal_inbin), ngal_thresh, replace=False) 
                weights[keep_ind] = 1./f_down

        f_down = np.float(len(catalog['halo.m'])) / np.float(np.sum(weights > 0.)) 
        print 'downsampled by ', int(round(f_down)), 'x'
        
        # ouptut downsampled catalog to hdf5 file  
        down_file = self.File(downsampled=str(int(round(f_down))))
        print 'writing to ... ', down_file
        f = h5py.File(down_file, 'w') 
        grp = f.create_group('data') 
        for key in catalog.keys(): 
            if key != 'weights': 
                grp.create_dataset(key, data=catalog[key])
        grp.create_dataset('weights', data=weights)
        f.close() 
        return None 


class Observations(object):
    '''
    '''
    def __init__(self, cat, **cat_kwargs):
        ''' Load in disparate observations into some universal format. 
        '''
        available = ['group_catalog']

        if cat not in available: 
            raise ValueError('must be one of the following : '+','.join(available))
        self.name = cat
    
        self._Unpack_kwargs(cat_kwargs)     # unpack keywoard arguments that specify the catalog 

    def Read(self):

        if self.name == 'group_catalog': 
            catalog = self._ReadGroupCat()
        else: 
            raise NotImplementedError

        return catalog 
    
    def _ReadGroupCat(self):
        '''
        '''
        # check that appropriate kwargs are set 
        assert 'Mrcut' in self._catalog_kwargs.keys()
        assert 'masscut' in self._catalog_kwargs.keys()
        assert 'position' in self._catalog_kwargs.keys()

        h = 0.7
        # load in galdata .fits file  
        galdata_file = ''.join([UT.dat_dir(), 'observations/', 
            'clf_groups_M', str(self._catalog_kwargs['Mrcut']), 
            '_', str(self._catalog_kwargs['masscut']), '_D360.', 
            'galdata_corr.fits']) 
        gal_data = mrdfits(galdata_file) 
    
        # process some of the data
        for column in gal_data.__dict__.keys(): 
            column_data = getattr(gal_data, column)
            if column == 'stellmass': # stellmass is in units of Msol/h^2
                column_data = column_data / h**2
                # convert to log Mass
                setattr(gal_data, 'mass', np.log10(column_data))
            elif column == 'ssfr': 
                column_data = column_data + np.log10(h**2)
                # convert to log Mass 
                setattr(gal_data, 'ssfr', column_data)    
            elif column == 'cz': # convert to z else: 
                setattr(gal_data, 'z', column_data/299792.458)
            else: 
                pass
        setattr(gal_data, 'sfr', gal_data.mass + gal_data.ssfr)     # get sfr values
        
        # read in central/satellite probability file 
        file = ''.join([UT.dat_dir(), 'observations/', 
            'clf_groups_M', str(self._catalog_kwargs['Mrcut']), 
            '_', str(self._catalog_kwargs['masscut']), '_D360.', 
            'prob.fits']) 
        prob_data = mrdfits(file)            

        # central or satellite 
        if self._catalog_kwargs['position'] == 'central': 
            prob_index = np.where(prob_data.p_sat <= 0.5)
        elif self._catalog_kwargs['position'] == 'satellite': 
            prob_index = np.where(prob_data.p_sat > 0.5)
        elif self._catalog_kwargs['position'] == 'all': 
            prob_index = range(len(prob_data.p_sat))
        else: 
            raise ValueError
    
        # hardcoded list of columns... but oh well 
        columns = ['ra', 'dec', 'mass', 'sfr', 'ssfr', 'z']
        
        # save columns into catalog dictionary 
        catalog = {} 
        for column in columns: 
            column_data = getattr(gal_data, column)[prob_index]
            if column in ['ra', 'dec']: 
                catalog[column] = column_data * 57.2957795
            else: 
                catalog[column] = column_data
        
        return catalog 
        
    def _Unpack_kwargs(self, kwargs): 
        ''' Process the keyword arguments for different observable 
        catalogs. And load to self._catalog_kwargs
        '''
        if self.name == 'group_catalog': 
            # absolute magnitude and mass cuts 
            Mrcut = kwargs['Mrcut']
            if Mrcut == 18:
                masscut='9.4'
            elif Mrcut == 19: 
                masscut='9.8'
            elif Mrcut == 20: 
                masscut='10.2'
            
            # save to object
            self._catalog_kwargs = {}
            self._catalog_kwargs['Mrcut'] = Mrcut
            self._catalog_kwargs['masscut'] = masscut 
            self._catalog_kwargs['position'] = kwargs['position']

            

"""
    def Build_CentralTree(nsnap_ancestor=20): 
        ''' Find descendants by tracking parents and children through TreePM's halos. 
        
        For snapshot 1 *pure* central subhalos, get the following: 
        - SHAMed M* for snapshots 2 to nsnap_ancestor
        - M_halo for snapshots 2 to nsnap_ancestor
        - M_halo,max for snapshots 2 to nsnap_ancestor

        ''' 
        # first import all the central subhalos 
        central_SHaloSHAM_file = lambda n_snap: ''.join([ 
            '/data1/hahn/wetzel_tree/dat/wetzel_tree/', 
            'subhalo_sham.central.snapshot', str(n_snap), 
            '.ancestor', str(nsnap_ancestor), 
            '.scatter0.2.li-march.hdf5']) 
        for i_snap in range(1,nsnap_ancestor+1): # check that everything exists
            if not os.path.isfile(central_SHaloSHAM_file(i_snap)): 
                raise ValueError
        
        # import all SHAMed central subhalos 
        central_SHaloSHAM_list = [] 
        for i_snap in range(1,nsnap_ancestor): 
            f = h5py.File(central_SHaloSHAM_file(i_snap))
            grp = f['cenque_data'] 

            cenSHaloSHAM = {'snapshot': i_snap} 
            for key in grp.keys(): 
                cenSHaloSHAM[key] = grp[key].value()
            f.close() 
            central_SHaloSHAM_list.append(cenSHaloSHAM)
        print 'All snapshots available!'
        print '------------------------'

        # Ancestor snapshot
        f = h5py.File(central_SHaloSHAM_file(nsnap_ancestor))
        grp = f['cenque_data'] 
        cenSHaloSHAM = {'snapshot': i_snap} 
        for key in grp.keys(): 
            cenSHaloSHAM[key] = grp[key].value()
        f.close() 
        ancestor = cenSHaloSHAM

        # go down the snapshots from nsnap_ancestor and track subhalos
        for i_snap in range(1, self.nsnap_ancestor)[::-1]:    
            cenSHaloSHAM = central_SHaloSHAM_list[i_snap]
            ancestor_index = cenSHaloSHAM['ancestor'+str(nsnap_ancestor)]

            # has ancestors at nsnap_ancestor
            has_ancestor, has_descendant = Util.intersection_index(
                    ancestor_index, ancestor['snap_index'])
            print ''
            print 'Snanpshot ', i_snap
            print len(has_ancestor), ' of ', len(child.snap_index), ' galaxies have ancestor at', nsnap_ancestor
            print '------------------------'

            # save SHAM and Halo masses
            Msham = np.repeat(-999., len(ancestor['snap_index']))
            Mhalo = np.repeat(-999., len(ancestor['snap_index']))
            Msham[has_descendant] = cenSHaloSHAM['mass'][has_ancestor]
            Mhalo[has_descendant] = cenSHaloSHAM['halo_mass'][has_ancestor]

            # snapshot, t_cosmic, and redshift where the subhalo starts 
            # hosting a galaxy. Aslo the mass of the new galaxy
            nsnap_genesis = np.repeat(-999, len(child.snap_index))
            mass_genesis = np.repeat(-999., len(child.snap_index)) 
            halomass_genesis = np.repeat(-999., len(child.snap_index)) 
            ancs = ancestor_index[has_ancestor] # ancestor indices

            # go through higher redshift snapshots in order to determine when the
            # subhalo was first "started"
            for ii_snap in range(i_snap, self.nsnap_ancestor): 
                
                child_ii = child_list[ii_snap-1]
                ii_anc, anc_ii = Util.intersection_index(getattr(child_ii, 'ancestor'+str(self.nsnap_ancestor)), ancs)
                
                massive = np.where(child_ii.mass[ii_anc] > 0.0)

                nsnap_genesis[has_ancestor[anc_ii[massive]]] = ii_snap
                mass_genesis[has_ancestor[anc_ii[massive]]] = child_ii.mass[ii_anc[massive]]
                halomass_genesis[has_ancestor[anc_ii[massive]]] = child_ii.halo_mass[ii_anc[massive]]
                anc_nsnap_genesis[has_descendant[anc_ii[massive]]] = ii_snap
                anc_mass_genesis[has_descendant[anc_ii[massive]]] = child_ii.mass[ii_anc[massive]]
                anc_halomass_genesis[has_descendant[anc_ii[massive]]] = child_ii.halo_mass[ii_anc[massive]]

            massive_ancestor = np.where(self.ancestor.mass[has_descendant] > 0.0)
            nsnap_genesis[has_ancestor[massive_ancestor]] = self.nsnap_ancestor 
            mass_genesis[has_ancestor[massive_ancestor]] = self.ancestor.mass[has_descendant[massive_ancestor]]
            halomass_genesis[has_ancestor[massive_ancestor]] = self.ancestor.halo_mass[has_descendant[massive_ancestor]]
            anc_nsnap_genesis[has_descendant[massive_ancestor]] = self.nsnap_ancestor 
            anc_mass_genesis[has_descendant[massive_ancestor]] = self.ancestor.mass[has_descendant[massive_ancestor]]
            anc_halomass_genesis[has_descendant[massive_ancestor]] = self.ancestor.halo_mass[has_descendant[massive_ancestor]]

            nonneg = np.where(nsnap_genesis[has_ancestor] > 0)
            tsnap_genesis = np.repeat(-999., len(nsnap_genesis))
            zsnap_genesis = np.repeat(-999., len(nsnap_genesis))
            tsnap_genesis[has_ancestor[nonneg]] = Util.get_t_nsnap(nsnap_genesis[has_ancestor[nonneg]])
            zsnap_genesis[has_ancestor[nonneg]] = Util.get_z_nsnap(nsnap_genesis[has_ancestor[nonneg]])

            print 'Children with ancestors ', len(has_ancestor[nonneg]), \
                    ' All children ', len(child.mass)
            print 'Greater than 10^10, Children with ancestors ', np.sum(child.mass[has_ancestor[nonneg]] > 10.), \
                    ' All children ', np.sum(child.mass > 10.), np.float(np.sum(child.mass[has_ancestor[nonneg]] > 10.))/np.float(np.sum(child.mass > 10.))
            print 'Greater than 10^9.5, Children with ancestors ', np.sum(child.mass[has_ancestor[nonneg]] > 9.5), \
                    ' All children ', np.sum(child.mass > 9.5), np.float(np.sum(child.mass[has_ancestor[nonneg]] > 9.5))/np.float(np.sum(child.mass > 9.5))

            #neg = np.where(nsnap_massive[has_ancestor] < 0)
            #print child.mass[has_ancestor[neg]]

            # trim sample to only keep galaxies that have ancestors at nsnap_ancestor and 
            # and 'starts' before nsnap.
            child.sample_trim(has_ancestor[nonneg])  
            setattr(child, 'nsnap_genesis', nsnap_genesis[has_ancestor[nonneg]])
            setattr(child, 'tsnap_genesis', tsnap_genesis[has_ancestor[nonneg]])
            setattr(child, 'zsnap_genesis', zsnap_genesis[has_ancestor[nonneg]])
            setattr(child, 'mass_genesis', mass_genesis[has_ancestor[nonneg]])
            setattr(child, 'halomass_genesis', halomass_genesis[has_ancestor[nonneg]])
            child.data_columns += ['nsnap_genesis', 'tsnap_genesis', 'zsnap_genesis', 'mass_genesis', 'halomass_genesis']

            setattr(self, 'descendant_snapshot'+str(i_snap), child)

        positive = np.where(anc_nsnap_genesis > 0)
        self.ancestor.sample_trim(positive[0])
        self.ancestor.data_columns += ['nsnap_genesis', 'tsnap_genesis', 'zsnap_genesis', 'mass_genesis', 'halomass_genesis', 'Msham_evol', 'Mhalo_evol']

        anc_tsnap_genesis[positive] = Util.get_t_nsnap(anc_nsnap_genesis[positive])
        anc_zsnap_genesis[positive] = Util.get_z_nsnap(anc_nsnap_genesis[positive])

        setattr(self.ancestor, 'nsnap_genesis', anc_nsnap_genesis[positive])
        setattr(self.ancestor, 'tsnap_genesis', anc_tsnap_genesis[positive])
        setattr(self.ancestor, 'zsnap_genesis', anc_zsnap_genesis[positive])
        setattr(self.ancestor, 'mass_genesis', anc_mass_genesis[positive])
        setattr(self.ancestor, 'halomass_genesis', anc_halomass_genesis[positive])
        setattr(self.ancestor, 'Msham_evol', anc_Msham[positive, :])
        setattr(self.ancestor, 'Mhalo_evol', anc_Mhalo[positive, :])

        return None
"""
