'''



'''
import time
import h5py
import numpy as np
from scipy import interpolate
#from scipy.integrate import odeint

# --- local --- 
import util as UT
import sfrs 
import observables as obvs


class GalPop(object): 
    def __init__(self): 
        ''' Empty class object for galaxy catalogs 
        '''
        pass 


def CenQue_File(type, cenque='default', downsampled=None): 
    ''' Function for getting the file name of CenQue Files when cenque and type are 
    specified. 
    '''
    if cenque == 'default': 
        tf = 7 
        abcrun = 'RHOssfrfq_TinkerFq_Std'
        prior = 'updated'
    elif cenque == 'nosmfevol':
        tf = 8 
        abcrun = 'RHOssfrfq_TinkerFq_NOSMFevol'
        prior = 'updated'
    else: 
        raise NotImplementedError
    
    # cenque files
    if type == 'sfms': 
        galpop_str = 'sfms'
    elif type == 'quenched': 
        galpop_str = 'quenched'
    else: 
        raise ValueError
    if downsampled is None: 
        down_str = ''
    else: 
        down_str = ''.join(['.down', str(downsampled), 'x']) 

    file = ''.join([UT.dat_dir(), 'cenque/',
        galpop_str, '.centrals.', 
        'tf', str(tf), 
        '.abc_', abcrun, 
        '.prior_', prior, 
        down_str, 
        '.hdf5']) 
    return file 


def Read_CenQue(type, cenque='default', downsampled=None):
    ''' Read in either (SF and Quenching galaixes) or (Quenched galaxies)
    generated from the CenQue project. 
    '''
    file = CenQue_File(type, cenque=cenque, downsampled=downsampled)
    gpop = GalPop()

    # read in the file and save to object
    f = h5py.File(file, 'r')  
    grp = f['data'] 
    for col in grp.keys(): 
        if col == 'mass': 
            # make sure to mark as SHAM mass
            if type == 'sfms': 
                setattr(gpop, 'M_sham', grp[col][:])    
            elif type == 'quenched': 
                setattr(gpop, 'M_sham', grp[col][:])    
                setattr(gpop, 'mass', grp[col][:])    
        elif col in ['sfr', 'ssfr']:
            continue 
        else: 
            setattr(gpop, col, grp[col][:])

    for key in grp.attrs.keys(): 
        if key in ['sfms', 'tau']: 
            attr_dict = {}
            for keyind in grp.attrs[key].split(','): 
                try: 
                    attr_dict[keyind.split(':')[0]] = float(keyind.split(':')[1])
                except ValueError:
                    attr_dict[keyind.split(':')[0]] = keyind.split(':')[1]

            setattr(gpop, key+'_dict', attr_dict)
        else:
            setattr(gpop, key+'_attr', grp.attrs[key])

    f.close() 
        
    # some small pre-processing here so convenience 
    setattr(gpop, 'zsnap_genesis', UT.z_from_t(gpop.tsnap_genesis)) 

    return gpop 


def DownsampleCenQue(cenque='default', ngal_thresh=4000, dmhalo=0.2): 
    ''' Downsample the CenQue objects in order to make the calculations more 
    tractable. Bin the halos in terms of their z = 0 masses and down sample 
    them so that there is still sufficient statistics in each bin.  
    '''
    # read in the entire galaxy population  
    q_pop = CentralQuenched(cenque=cenque, downsampled=None)
    print q_pop._File() 
    q_pop._Read_CenQue()
    sfms_pop = CentralMS(cenque=cenque, downsampled=None)
    print sfms_pop._File() 
    sfms_pop._Read_CenQue()
    
    # halo mass bins delta M_h = 0.2 dex for now... 
    mhalo_bin = np.arange(
            np.min([sfms_pop.halo_mass.min(), q_pop.halo_mass.min()]) - 0.5 * dmhalo, 
            np.max([sfms_pop.halo_mass.max(), q_pop.halo_mass.max()]) + dmhalo, 
            dmhalo
            ) 
    
    # weights 
    weights_sf = np.repeat(1., len(sfms_pop.M_sham)) 
    weights_q = np.repeat(1., len(q_pop.M_sham)) 

    Ngal_sf, Ngal_q = 0, 0 
    for i_m in range(len(mhalo_bin) - 1): 
        inbin_sf = np.where(
                (mhalo_bin[i_m] < sfms_pop.halo_mass) &
                (mhalo_bin[i_m+1] >= sfms_pop.halo_mass)
                ) 
        inbin_q = np.where(
                (mhalo_bin[i_m] < q_pop.halo_mass) &
                (mhalo_bin[i_m+1] >= q_pop.halo_mass)
                ) 
        ngal_sf = len(inbin_sf[0]) 
        ngal_q = len(inbin_q[0]) 

        # make sure there's sufficient statistics for both star forming and quiescent
        # galaxies based on a rough criteria for now ... assign appropriate 
        # corresponding weights 
        if (ngal_sf + ngal_q > ngal_thresh):  
            f_down = ngal_thresh/np.float(ngal_sf + ngal_q) 
            #print np.float(ngal_q) * f_down

            weights_q[inbin_q] = 0.
            weights_sf[inbin_sf] = 0.

            keep_ind = np.random.choice(range(ngal_sf + ngal_q), ngal_thresh, replace=False) 
            keep_sf = (inbin_sf[0])[keep_ind[np.where(keep_ind < ngal_sf)]]
            keep_q = (inbin_q[0])[keep_ind[np.where(keep_ind >= ngal_sf)] - ngal_sf]

            weights_sf[keep_sf] = 1./f_down
            weights_q[keep_q] = 1./f_down
    
            # sanity check
            if (int(round(np.sum(weights_sf[inbin_sf]) + np.sum(weights_q[inbin_q]))) 
                    != ngal_sf+ngal_q): 
                print np.sum(weights_sf[inbin_sf]) + np.sum(weights_q[inbin_q])
                print ngal_sf+ngal_q 
                raise ValueError
         
        Ngal_sf += ngal_sf
        Ngal_q += ngal_q
        
    # More sanity checks
    if int(round(np.sum(weights_sf) + np.sum(weights_q))) != len(sfms_pop.M_sham)+len(q_pop.M_sham): 
        print np.sum(weights_sf) + np.sum(weights_q), len(sfms_pop.M_sham)+len(q_pop.M_sham)
        raise ValueError

    # the galaxy sample is downsamped by
    down_frac = np.float(Ngal_sf + Ngal_q) / \
            np.float(len(weights_sf[np.where(weights_sf != 0.)]) + 
                    len(weights_q[np.where(weights_q != 0.)])) 
    print 'downsampled ', int(round(down_frac)), 'x'
        
    # save the downsampled galaxy populations to file
    q_file = h5py.File(q_pop._File(), 'r') 
    sf_file = h5py.File(sfms_pop._File(), 'r') 
    
    q_down_file = h5py.File(q_pop._File().replace(
        '.hdf5', ''.join(['.down', str(int(round(down_frac))), 'x', '.hdf5'])), 'w') 
    sf_down_file = h5py.File(sfms_pop._File().replace(
        '.hdf5', ''.join(['.down', str(int(round(down_frac))), 'x', '.hdf5'])), 'w') 
    q_file.copy('data', q_down_file) 
    sf_file.copy('data', sf_down_file) 
    q_file.close() 
    sf_file.close() 

    insample_q = np.where(weights_q != 0.) 
    insample_sf = np.where(weights_sf != 0.) 

    q_grp = q_down_file['data']
    for col in q_grp.keys():
        new_col = q_grp[col].value[insample_q]
        q_grp.__delitem__(col) 
        q_grp.create_dataset(col, data = new_col) 
    q_grp.create_dataset('weight_down', data = weights_q[insample_q])
    q_down_file.close()  

    sf_grp = sf_down_file['data']
    for col in sf_grp.keys():
        new_col = sf_grp[col].value[insample_sf]
        sf_grp.__delitem__(col) 
        sf_grp.create_dataset(col, data = new_col) 
    sf_grp.create_dataset('weight_down', data = weights_sf[insample_sf])
    sf_down_file.close()  

    return None


class CentralQuenched(GalPop):  # Quenched Central Galaxies
    def __init__(self, cenque='default', downsampled=20):
        ''' This object reads in the quenched galaxies generated 
        from the CenQue project and is an object for those galaxies. 
        '''
        self.cenque = cenque
        self.downsampled = downsampled
        self.mass = None
        self.sfr = None
        self.ssfr = None 

    def _Read_CenQue(self):  
        galpop = Read_CenQue('quenched', cenque=self.cenque, downsampled=self.downsampled)
        for key in galpop.__dict__.keys(): 
            setattr(self, key, getattr(galpop, key))
        return None 

    def _File(self): 
        return CenQue_File('quenched', cenque=self.cenque, downsampled=self.downsampled)


class CentralMS(GalPop):        # Star-forming + Quenching Central Galaxies
    def __init__(self, cenque='default', downsampled=20):
        ''' This object reads in the star-forming and quenching
        galaxies generated from the CenQue project and is an object
        for those galaxies. Unlike CenQue, this object WILL NOT
        have extensive functions and will act as a data catalog. 
     
        '''
        self.cenque = cenque
        self.downsampled = downsampled
        self.mass = None
        self.sfr = None
        self.ssfr = None 
        self.sfr_genesis = None

    def _Read_CenQue(self):  
        ''' Read in SF and Quenching galaixes generated from 
        the CenQue project. 
        '''
        galpop = Read_CenQue('sfms', cenque=self.cenque, downsampled=self.downsampled)
        for key in galpop.__dict__.keys(): 
            setattr(self, key, getattr(galpop, key))

        return None 

    def _File(self): 
        return CenQue_File('sfms', cenque=self.cenque, downsampled=self.downsampled)


class Evolver(object): 
    def __init__(self, cms, evol_dict=None): 
        ''' Class object that evolves the CentralMS galaxy catalog catalog object .
        Object contains suite of functions for the evolution. 
        '''
        self.cms = cms 
        if evol_dict is None:  # default 
            raise ValueError
        else: 
            self.evol_dict = evol_dict

    def __call__(self):  
        self.Evolve()
        return self.cms 


    def Evolve(self): 
        ''' Evolve SFR and calculated integrated SFR stellar masses. 
        The module creates lambda functions for log SFR(t) and then 
        integrates that. Currently set up to minimize the amount of
        specific cases. 
        '''
        # ---- 
        # construct z(t) interpolation function to try to speed up the integration 
        z_table, t_table = UT.zt_table()     
        z_of_t = interpolate.interp1d(list(reversed(t_table)), list(reversed(z_table)), 
                kind='cubic') 
        t_output = t_table[1:16][::-1]
        tsnaps = t_table[1:16]

        # ---- 
        # different SFH prescriptions
        # ---- 
        sfh_kwargs = {'name': self.evol_dict['sfh']['name'], 'sfms': self.cms.sfms_dict} 
        if self.evol_dict['sfh']['name'] == 'constant_offset': 
            if self.evol_dict['sfh']['assembly_bias'] == 'none': 
                # No assembly bias. No dutycycle
                sigma_logsfr = sfrs.ScatterLogSFR_sfms(
                        self.cms.mass_genesis, self.cms.zsnap_genesis, 
                        sfms_dict=self.cms.sfms_dict)
                sfh_kwargs['dsfr'] = sigma_logsfr * np.random.randn(len(self.cms.mass_genesis))

            elif self.evol_dict['sfh']['assembly_bias'] == 'longterm': 
                # long term assembly bias (rank ordered by the ultimate mass growth) 
                # scatter determines the scatter in the assembly bias 
                # scatter = 0.3 is the same as none
                dMhalo = self.cms.halo_mass -  self.cms.halomass_genesis

                mu_logsfr = sfrs.AverageLogSFR_sfms(self.cms.mass_genesis, self.cms.zsnap_genesis, 
                        sfms_dict=self.cms.sfms_dict)
                sigma_logsfr = sfrs.ScatterLogSFR_sfms(self.cms.mass_genesis, self.cms.zsnap_genesis,
                        sfms_dict=self.cms.sfms_dict)
                if self.evol_dict['sfh']['sigma_bias'] > sigma_logsfr: 
                    raise ValueError("You can't have negative scatter!") 

                sigma_eff = np.sqrt(sigma_logsfr**2 - self.evol_dict['sfh']['sigma_bias']**2)
                dsfr = sigma_eff * np.random.randn(len(mu_logsfr)) 
                dsfr_scat = self.evol_dict['sfh']['sigma_bias'] * np.random.randn(len(mu_logsfr))
    
                dsfr_tot = np.repeat(-999., len(dsfr))
                self.cms.sfr_genesis = np.zeros(len(mu_logsfr))
                for tg in np.unique(self.cms.tsnap_genesis): 
                    snap = np.where(self.cms.tsnap_genesis == tg)[0]

                    M_h_bins = np.arange(self.cms.halomass_genesis[snap].min()-0.1, 
                            self.cms.halomass_genesis[snap].max()+0.2, 0.2)
                    for im in xrange(len(M_h_bins)-1): 
                        inbin = np.where(
                                (self.cms.halomass_genesis[snap] > M_h_bins[im]) & 
                                (self.cms.halomass_genesis[snap] <= M_h_bins[im+1])
                                )[0]
                        Mhsort = np.argsort(dMhalo[snap[inbin]]) 
                        self.cms.sfr_genesis[snap[inbin[Mhsort]]] = \
                                (mu_logsfr[snap])[inbin[Mhsort]] + \
                                np.sort(dsfr[snap[inbin]]) + dsfr_scat[snap[inbin]]
                        dsfr_tot[snap[inbin[Mhsort]]] = np.sort(dsfr[snap[inbin]]) + dsfr_scat[snap[inbin]]
                sfh_kwargs['dsfr'] = dsfr_tot 

        elif self.evol_dict['sfh']['name'] == 'random_step': 
            if self.evol_dict['sfh']['assembly_bias'] == 'none': 
                # No assembly bias. 
                # Random step function duty cycle 
                del_t_max = 13.1328 - self.cms.tsnap_genesis.min() 
                
                # the frequency range of the steps 
                tshift_min = self.evol_dict['sfh']['dt_min'] 
                tshift_max = self.evol_dict['sfh']['dt_max'] 

                n_col = int(np.ceil(del_t_max/tshift_min))  # number of columns 
                n_gal = len(self.cms.mass_genesis)
                tshift = np.zeros((n_gal, n_col))
                tshift[:,1:] = np.random.uniform(tshift_min, tshift_max, size=(n_gal, n_col-1))
                sfh_kwargs['tshift'] = np.cumsum(tshift , axis=1) + \
                        np.tile(self.cms.tsnap_genesis, (n_col, 1)).T
                outofrange = np.where(sfh_kwargs['tshift'] > 13.1328)
                sfh_kwargs['tshift'][outofrange] = -999.
                sfh_kwargs['amp'] = np.random.randn(n_gal, n_col) *\
                        self.evol_dict['sfh']['sigma']

            elif self.evol_dict['sfh']['assembly_bias'] == 'acc_hist': 
                # Assembly bias from accretion history of halo. 
                # Random step function duty cycle 
                del_t_max = 13.1328 - self.cms.tsnap_genesis.min() 
                
                # the frequency range of the steps 
                tshift_min = self.evol_dict['sfh']['dt_min'] 
                tshift_max = self.evol_dict['sfh']['dt_max'] 

                n_col = int(np.ceil(del_t_max/tshift_min))  # number of columns 
                n_gal = len(self.cms.mass_genesis)
                tshift = np.zeros((n_gal, n_col))
                tshift[:,1:] = np.random.uniform(tshift_min, tshift_max, size=(n_gal, n_col-1))
                sfh_kwargs['tshift'] = np.cumsum(tshift , axis=1) + \
                        np.tile(self.cms.tsnap_genesis, (n_col, 1)).T
                outofrange = np.where(sfh_kwargs['tshift'] > 13.1328)
                sfh_kwargs['tshift'][outofrange] = -999.
            
                # Fractional halo growth
                dMhalo = -1. * np.diff(self.cms.Mhalo_hist, axis=1)
                # halo mass accretion history 
                #if self.evol_dict['sfh']['halo_prop'] == 'frac': 
                #    # Fractional halo growth
                #    dMhalo = -1. * np.diff(self.cms.Mhalo_hist, axis=1)
                #elif self.evol_dict['sfh']['halo_prop'] == 'abs': 
                #    # absolute halo growth 
                #    #dMhalo = -1. * np.diff(self.cms.Mhalo_hist, axis=1)
                #elif self.evol_dict['sfh']['halo_prop'] == 'longterm': 
               
                # scatter in the correlation between accretion history and SFR 
                # if sig_noise = 0.3, that means that accretion history and SFR
                # have nothing to do with each other; if sig_noise = 0.0, then 
                # SFR is rank ordered in accretion history 
                sig_noise = self.evol_dict['sfh']['sigma_bias']
                sig_eff = np.sqrt(
                        self.evol_dict['sfh']['sigma']**2 - 
                        self.evol_dict['sfh']['sigma_bias']**2
                        ) 

                # calculate the biased SFR 
                # (rank order dSFR based on the accretion history of the halo) 
                biased_dsfr = np.tile(-999., dMhalo.shape)
                for i_d in xrange(dMhalo.shape[1]): 
                    halo_exists = np.where(
                            (self.cms.Mhalo_hist[:,i_d] != -999.) &
                            (self.cms.Mhalo_hist[:,i_d+1] != -999.))[0]
                    #M_h_tip1 = self.cms.Mhalo_hist[:,i_d][halo_exists]
                    M_h_ti = self.cms.Mhalo_hist[:,i_d+1][halo_exists]
                    
                    M_h_bins = np.arange(M_h_ti.min()-0.1, M_h_ti.max()+0.2, 0.2) 
                    for im in xrange(len(M_h_bins)-1): 
                        inbin = np.where(
                                (M_h_ti > M_h_bins[im]) & 
                                (M_h_ti <= M_h_bins[im+1])
                                )[0]
                        sorted = np.argsort(dMhalo[:,i_d][halo_exists[inbin]])

                        biased_dsfr[:,i_d][halo_exists[inbin[sorted]]] = \
                                np.sort(np.random.randn(len(sorted))) * sig_eff
                    
                    #no_halo = np.logical_not(
                    #        (self.cms.Mhalo_hist[:,i_d] != -999.) &
                    #        (self.cms.Mhalo_hist[:,i_d+1] != -999.))
                    no_halo = np.where(
                            (self.cms.Mhalo_hist[:,i_d] == -999.) |
                            (self.cms.Mhalo_hist[:,i_d+1] == -999.))
                    biased_dsfr[:,i_d][no_halo] = np.random.randn(len(no_halo[0])) * sig_eff
                
                #print biased_dsfr.min(),  biased_dsfr.max()
                if biased_dsfr.min() == -999.: 
                    raise ValueError

                sfh_kwargs['amp'] = np.tile(-999., (n_gal, n_col)) 
                for i_t in xrange(len(tsnaps)-1): 
                    if i_t != len(tsnaps) - 2: 
                        t_range = np.where( 
                                (sfh_kwargs['tshift'] <= tsnaps[i_t]) &
                                (sfh_kwargs['tshift'] > tsnaps[i_t+1])
                                )
                    else: 
                        t_range = np.where( 
                                (sfh_kwargs['tshift'] <= tsnaps[i_t]) &
                                (sfh_kwargs['tshift'] >= tsnaps[i_t+1])
                                )

                    sfh_kwargs['amp'][t_range] = (biased_dsfr[:,i_t])[t_range[0]] +\
                            sig_noise * np.random.randn(len(t_range[0]))

                #test = np.where(
                #        (sfh_kwargs['tshift'] <= tsnaps[0]) & 
                #        (sfh_kwargs['tshift'] >= tsnaps[-1]) 
                #        ) 
                #print sfh_kwargs['amp'][test].min(),  sfh_kwargs['amp'][test].max()
                # ---- 
        # identify the quenching galaxies 
        qing = np.where((self.cms.t_quench != 999.) & (self.cms.t_quench > 0.))  
        t_final = np.repeat(13.1328, len(self.cms.mass_genesis))    
        t_final[qing] = self.cms.t_quench[qing]
    
        # ---- 
        # integrated SFR stellar mass while on SFMS 
        # for galaxies that do not quench, this is their final mass  
        # ---- 
        func_kwargs = {
                't_initial': self.cms.tsnap_genesis, 
                't_final': t_final, 
                'f_retain': self.evol_dict['mass']['f_retain'], 
                'zfromt': z_of_t, 
                'sfh_kwargs': sfh_kwargs
                }
        if self.evol_dict['mass']['type'] == 'rk4':     # RK4
            f_ode = sfrs.ODE_RK4
        elif self.evol_dict['mass']['type'] == 'euler': # Forward euler
            f_ode = sfrs.ODE_Euler
        integ_logM = f_ode(
                sfrs.dlogMdt_MS,                    # dy/dt
                self.cms.mass_genesis,              # logM0
                t_output,                # output times  
                self.evol_dict['mass']['t_step'],   # time step
                **func_kwargs) 
    
        self.cms.mass = (integ_logM.T)[:,-1].copy()     # integrated SFR mass 
        self.cms.sfr = sfrs.LogSFR_sfms(integ_logM[-1], z_of_t(t_final), sfms_dict=sfh_kwargs) 

        t_matrix = np.tile(t_output, (integ_logM.shape[1],1))
        t_genesis_matrix = np.tile(self.cms.tsnap_genesis, (integ_logM.shape[0],1)).T
        t_final_matrix = np.tile(t_final, (integ_logM.shape[0],1)).T

        outofbounds = np.where((t_matrix < t_genesis_matrix) | (t_matrix > t_final_matrix))
        self.cms.Minteg_hist = integ_logM.T.copy()
        self.cms.Minteg_hist[outofbounds] = -999.       
        
        # ---- 
        # quenching galaxies after they're off the SFMS
        t_final = np.repeat(13.1328, len(self.cms.mass_genesis))
        tauQ = sfrs.getTauQ(self.cms.mass[qing], tau_dict=self.cms.tau_dict)

        # calculate finall stellar mass of quenching galaxies
        t_output = np.array(
                [self.cms.t_quench[qing].min()] +
                list(t_output[np.where(t_output > self.cms.t_quench[qing].min())])
                )
        func_kwargs = {
                'logSFR_Q': self.cms.sfr[qing],         # log SFR_Q
                'tau_Q': tauQ,
                't_Q': self.cms.t_quench[qing],         # quenching time 
                'f_retain': self.evol_dict['mass']['f_retain'], 
                't_final': t_final[qing]
                }
        integ_logM_Q = f_ode(
                sfrs.dlogMdt_Q, 
                self.cms.mass[qing], 
                t_output,
                self.evol_dict['mass']['t_step'],   # time step
                **func_kwargs)
        self.cms.mass[qing] = integ_logM_Q.T[:,-1].copy() 
        self.cms.sfr[qing] = sfrs.LogSFR_Q(
                13.1328, 
                logSFR_Q=self.cms.sfr[qing],         # log SFR_Q
                tau_Q=tauQ,
                t_Q=self.cms.t_quench[qing])
        
        t_quench_matrix = np.tile(self.cms.t_quench, (integ_logM.shape[0],1)).T
        qqing = np.where(
                (t_quench_matrix != 999.) & (t_quench_matrix > 0.) & 
                (t_matrix >= t_quench_matrix)) 
        t_q_matrix = np.tile(self.cms.t_quench[qing], (integ_logM_Q.shape[0]-1,1)).T
        tt_matrix = np.tile(t_output[1:], (integ_logM_Q.shape[1],1))
        qqqing = np.where(tt_matrix >= t_q_matrix)
        
        self.cms.Minteg_hist[qqing] = integ_logM_Q.T[qqqing].copy() 
        self.cms.sfh_dict = sfh_kwargs
        return None


class EvolvedGalPop(GalPop): 
    def __init__(self, cenque='default', evol_dict=None, downsampled=20): 
        ''' Class object for the (integrated SFR) evolved galaxy population. 
        Convenience functions for saving and reading the files so that it does
        not have to be run over and over again. Currently the hdf5 files include
        only the bare minimum metadata. 
        '''
        self.cenque = cenque
        self.downsampled = downsampled
        self.evol_dict = evol_dict

    def File(self): 
        '''
        '''
        # Write Star forming and Quenching catalog 
        evol_file = ''.join([UT.dat_dir(), 'galpop/' 
            'sfms.centrals', 
            self._Spec_str(),
            '.hdf5'])
        return evol_file 

    def _Spec_str(self): 
        spec_str = ''.join([
            '.', self._CenQue_str(), self._SFH_str(), self._Mass_str()
            ])
        return spec_str

    def _CenQue_str(self): 
        if self.downsampled is None: 
            down_str = ''
        else: 
            down_str = ''.join(['.down', str(self.downsampled), 'x'])
        #cq_str = ''.join(['tf', str(tf), '.abc_', abcrun, '.prior_', prior])
        cq_str = ''.join([self.cenque, down_str] )
        return cq_str

    def _SFH_str(self): 
        '''
        '''
        if self.evol_dict['sfh']['name'] in ['constant_offset', 'no_scatter']: 
            if self.evol_dict['sfh']['assembly_bias'] == 'none': 
                sfh_str = ''.join([
                    '.SFH', self.evol_dict['sfh']['name'], 
                    '.AsBias', self.evol_dict['sfh']['assembly_bias']])
            elif self.evol_dict['sfh']['assembly_bias'] == 'longterm': 
                sfh_str = ''.join([
                    '.SFH', self.evol_dict['sfh']['name'], 
                    '.AsBias', self.evol_dict['sfh']['assembly_bias'], 
                    '_sigma', str(round(self.evol_dict['sfh']['sigma_bias'],1))])

        elif self.evol_dict['sfh']['name'] in ['random_step']: 
            if self.evol_dict['sfh']['assembly_bias'] == 'none': 
                sfh_str = ''.join([
                    '.SFH', self.evol_dict['sfh']['name'],
                    '_dt', str(round(self.evol_dict['sfh']['dt_min'],2)),
                    '_', str(round(self.evol_dict['sfh']['dt_max'],2)),
                    '_sigma', str(round(self.evol_dict['sfh']['sigma'],1))
                    ])
            elif self.evol_dict['sfh']['assembly_bias'] == 'acc_hist': 
                sfh_str = ''.join([
                    '.SFH', self.evol_dict['sfh']['name'],
                    '_dt', str(round(self.evol_dict['sfh']['dt_min'],2)),
                    '_', str(round(self.evol_dict['sfh']['dt_max'],2)), 
                    '_sigma', str(round(self.evol_dict['sfh']['sigma'],1)),
                    '.AsBias', self.evol_dict['sfh']['assembly_bias'], 
                    '_sigma',  str(self.evol_dict['sfh']['sigma_bias'])
                    ])
        else: 
            raise ValueError
        return sfh_str 
    
    def _Mass_str(self): 
        mass_str = ''.join([ 
            '.M_', self.evol_dict['mass']['type'], 
            '_tstep', str(self.evol_dict['mass']['t_step']) 
            ])
        return mass_str

    def Write(self):  
        ''' Run the evolver on the CentralMS object with the 
        evol_dict specificiations and then save to file. 
        '''
        t_start = time.time() 
        cms = CentralMS(cenque=self.cenque, downsampled=self.downsampled)
        cms._Read_CenQue()
        eev = Evolver(cms, evol_dict=self.evol_dict)
        MSpop = eev()
        print time.time() - t_start, ' seconds to run the evolver'

        f = h5py.File(self.File(), 'w')    
        grp = f.create_group('data')
        # hardcoded main data columns for the catalogs
        for col in ['mass', 'sfr', 'halo_mass', 'M_sham', 
                'tsnap_genesis', 'nsnap_genesis', 'zsnap_genesis', 
                'mass_genesis', 'halomass_genesis', 
                't_quench', 'Minteg_hist', 'Msham_hist', 'Mhalo_hist', 
                'weight_down']: 
            grp.create_dataset(col, data = getattr(MSpop, col)) 
    
        # SFH dictionary 
        sfh_grp = f.create_group('sfh_dict')
        for key in MSpop.sfh_dict.keys(): 
            if key == 'sfms':   # SFMS dictinary
                sfms_prop_str = ','.join([   
                    ':'.join([k, str(MSpop.sfh_dict['sfms'][k])]) 
                    for k in MSpop.sfh_dict['sfms'].keys()
                    ])
                sfh_grp.create_dataset(key, data=str(sfms_prop_str))
            else: 
                if isinstance(MSpop.sfh_dict[key], str): 
                    sfh_grp.create_dataset(key, data=str(MSpop.sfh_dict[key]))
                else: 
                    sfh_grp.create_dataset(key, data=MSpop.sfh_dict[key]) 

        f.close()
        return None

    def Read(self): 
        ''' Read in the hdf5 file. 
        '''
        f = h5py.File(self.File(), 'r')    
        # read main data columns 
        grp = f['data']
        for key in grp.keys(): 
            setattr(self, key, grp[key][:])
        
        # read in SFH dictionary
        sfh_grp = f['sfh_dict']
        self.sfh_dict = {} 
        for key in sfh_grp.keys(): 
            if key == 'sfms': 
                self.sfh_dict[key] = {} 
                for keyind in (sfh_grp[key].value).split(','): 
                    try: 
                        self.sfh_dict[key][keyind.split(':')[0]] = float(keyind.split(':')[1])
                    except ValueError:
                        self.sfh_dict[key][keyind.split(':')[0]] = keyind.split(':')[1]
            else: 
                self.sfh_dict[key] = sfh_grp[key].value

        f.close()
        return None




if __name__=='__main__': 
    #EGP_FreqTest()
    DownsampleCenQue(cenque='nosmfevol') 
    # testing purposes
    #for scat in [0.0, 0.1, 0.2, 0.3]:
    #    evol_dict = {
    #            'sfh': {'name': 'random_step', 'dt_min': 0.1, 'dt_max':0.25, 'sigma': 0.3,
    #                'assembly_bias': 'acc_hist', 'halo_prop': 'frac', 'sigma_bias': scat}, 
    #            'mass': {'type': 'euler', 'f_retain': 0.6, 't_step': 0.1} 
    #            } 
    #    EGP = EvolvedGalPop(cenque='default', evol_dict=evol_dict)
    #    EGP.Write() 
    #'sfh': {'name': 'random_step', 'dt_min': 0.01, 'dt_max':0.25, 'sigma': 0.3, 
    #    'assembly_bias': 'acc_hist', 'sigma_bias':0.2}, 
    # 'sfh': {'name': 'random_step', 'sigma':0.3, 'dt_min': 0.1, 'dt_max':0.5}, 
    #cms = CentralMS()
    #cms._Read_CenQue()

    #blah = AssignSFR0(cms)
    #eev = Evolver(blah)
    #new_blah = eev()
