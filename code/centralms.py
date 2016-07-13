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


class GalPop(object): 
    def __init__(self): 
        ''' Empty class object for galaxy catalogs 
        '''
        pass 


class CentralQuenched(GalPop):  # Quenched Central Galaxies
    def __init__(self, cenque='default'):
        ''' This object reads in the quenched galaxies generated 
        from the CenQue project and is an object for those galaxies. 
        '''
        self.cenque = cenque
        self.mass = None
        self.sfr = None
        self.ssfr = None 

    def _Read_CenQue(self):  
        
        galpop = Read_CenQue('quenched', cenque='default')
        for key in galpop.__dict__.keys(): 
            setattr(self, key, getattr(galpop, key))
            
        return None 


class CentralMS(GalPop):        # Star-forming + Quenching Central Galaxies
    def __init__(self, cenque='default'):
        ''' This object reads in the star-forming and quenching
        galaxies generated from the CenQue project and is an object
        for those galaxies. Unlike CenQue, this object WILL NOT
        have extensive functions and will act as a data catalog. 
     
        '''
        self.cenque = cenque
        self.mass = None
        self.sfr = None
        self.ssfr = None 
        self.sfr_genesis = None

    def _Read_CenQue(self):  
        ''' Read in SF and Quenching galaixes generated from 
        the CenQue project. 
        '''
        galpop = Read_CenQue('sfms', cenque='default')
        for key in galpop.__dict__.keys(): 
            setattr(self, key, getattr(galpop, key))

        return None 


def Read_CenQue(type, cenque='default'):
    ''' Read in either (SF and Quenching galaixes) or (Quenched galaxies)
    generated from the CenQue project. 
    '''
    if cenque == 'default': 
        tf = 7 
        abcrun = 'RHOssfrfq_TinkerFq_Std'
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
    file = ''.join([UT.dat_dir(), 'cenque/',
        galpop_str, '.centrals.', 
        'tf', str(tf), 
        '.abc_', abcrun, 
        '.prior_', prior, 
        '.hdf5']) 

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


class Evolver(object): 
    def __init__(self, cms, evol_dict=None): 
        ''' Class object that evolves the CentralMS galaxy catalog catalog object .
        Object contains suite of functions for the evolution. 
        '''
        self.cms = cms 
        if evol_dict is None:  # default 
            raise ValueError
            #self.evol_dict = {
            #        'sfr': {'name': 'constant_offset'}, 
            #        'mass': {'type': 'euler', 'f_retain': 0.6, 't_step': 0.01} 
            #        }
        else: 
            self.evol_dict = evol_dict

    def __call__(self):  
        self.AssignSFR0()
        self.Evolve()
        return self.cms 

    def AssignSFR0(self): 
        ''' Assign initial SFRs to cms object based on tsnap_genesis 
        (time when the halo enters the catalog) and mass_genesis
        '''
        if 'tsnap_genesis' not in self.cms.__dict__.keys(): 
            # Most likely you did not read in CenQue catalog!
            raise ValueError

        # Assign SFR to star-forming galaxies 
        if self.evol_dict['initial']['assembly_bias'] == 'none': 
            # No assembly bias 
            mu_logsfr = sfrs.AverageLogSFR_sfms(self.cms.mass_genesis, self.cms.zsnap_genesis, 
                    sfms_dict=self.cms.sfms_dict)
            sigma_logsfr = sfrs.ScatterLogSFR_sfms(self.cms.mass_genesis, self.cms.zsnap_genesis, 
                    sfms_dict=self.cms.sfms_dict)
            self.cms.sfr_genesis = mu_logsfr + sigma_logsfr * np.random.randn(len(mu_logsfr))
        elif self.evol_dict['initial']['assembly_bias'] == 'longterm': 
            # long term assembly bias (rank ordered by the ultimate mass growth) 
            dMhalo = self.cms.halo_mass -  self.cms.halomass_genesis

            mu_logsfr = sfrs.AverageLogSFR_sfms(self.cms.mass_genesis, self.cms.zsnap_genesis, 
                    sfms_dict=self.cms.sfms_dict)
            sigma_logsfr = sfrs.ScatterLogSFR_sfms(self.cms.mass_genesis, self.cms.zsnap_genesis, 
                    sfms_dict=self.cms.sfms_dict)
            sigma_eff = np.sqrt(sigma_logsfr**2 - self.evol_dict['initial']['scatter']**2)
            dsfr = sigma_eff * np.random.randn(len(mu_logsfr)) 
            dsfr_scat = self.evol_dict['initial']['scatter'] * np.random.randn(len(mu_logsfr))

            self.cms.sfr_genesis = np.zeros(len(mu_logsfr))
            for tg in np.unique(self.cms.tsnap_genesis): 
                snap = np.where(self.cms.tsnap_genesis == tg)[0]
                Mhsort = np.argsort(dMhalo[snap]) 
                self.cms.sfr_genesis[snap[Mhsort]] = (mu_logsfr[snap])[Mhsort] + \
                        np.sort(dsfr[snap]) + dsfr_scat[snap]
                #print 't = ', tg
                #print dMhalo[snap]
                #print dMhalo[snap][Mhsort]
                #print Mhsort
                #print (mu_logsfr[snap])[Mhsort]
                #print np.sort(dsfr[snap]) 
                #print  (self.cms.sfr_genesis[snap])[Mhsort]
        else: 
            raise NotImplementedError

        return None 

    def Evolve(self): 
        ''' Evolve SFR and calculated integrated SFR stellar masses. 
        The module creates lambda functions for log SFR(t) and then 
        integrates that. Currently set up to minimize the amount of
        specific cases. 
        '''
        # ---- 
        # different SFH prescriptions
        # ---- 
        sfh_kwargs = {'name': self.evol_dict['sfh']['name'], 'sfms': self.cms.sfms_dict} 

        if self.evol_dict['sfh']['name'] == 'constant_offset': 
            # No assembly bias. No dutycycle
            dsfr = self.cms.sfr_genesis - sfrs.AverageLogSFR_sfms(
                    self.cms.mass_genesis, UT.z_from_t(self.cms.tsnap_genesis), 
                    sfms_dict=self.cms.sfms_dict)
            sfh_kwargs['dsfr'] = dsfr

        elif self.evol_dict['sfh']['name'] == 'random_step': 
            # No assembly bias. Random step function duty cycle 
            del_t_max = 13.1328 - self.cms.tsnap_genesis.min() 
            tshift_min, tshift_max = self.evol_dict['sfh']['dt_min'], self.evol_dict['sfh']['dt_max']  # hardcoded for now 

            n_col = int(np.ceil(del_t_max/tshift_min))  # number of columns 
            n_gal = len(self.cms.mass_genesis)
            tshift = np.zeros((n_gal, n_col))
            tshift[:,1:] = np.random.uniform(tshift_min, tshift_max, size=(n_gal, n_col-1))
            sfh_kwargs['tshift'] = np.cumsum(tshift , axis=1) + \
                    np.tile(self.cms.tsnap_genesis, (n_col, 1)).T
            outofrange = np.where(sfh_kwargs['tshift'] > 13.1328)
            sfh_kwargs['tshift'][outofrange] = -999.
            sfh_kwargs['amp'] = np.random.randn(n_gal, n_col) * self.evol_dict['sfh']['sigma']

        # ---- 
        # construct z(t) interpolation function to try to speed up the integration 
        z_table, t_table = UT.zt_table()     
        z_of_t = interpolate.interp1d(list(reversed(t_table)), list(reversed(z_table)), 
                kind='cubic') 

        t_output = t_table[1:16][::-1]

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
    def __init__(self, cenque='default', evol_dict=None): 
        ''' Class object for the (integrated SFR) evolved galaxy population. 
        Convenience functions for saving and reading the files so that it does
        not have to be run over and over again. Currently the hdf5 files include
        only the bare minimum metadata. 
        '''
        self.cenque = cenque
        if evol_dict is None: 
            self.evol_dict = {
                    'sfr': {'name': 'constant_offset'}, 
                    'mass': {'type': 'rk4', 'f_retain': 0.6, 't_step': 0.01} 
                    }
        else: 
            self.evol_dict = evol_dict

    def File(self): 
        '''
        '''
        # Write Star forming and Quenching catalog 
        evol_file = ''.join([
            '/data1/hahn/centralMS/galpop/', 
            'sfms.centrals.', 
            self._Spec_str(),
            '.hdf5'])
        return evol_file 

    def _Spec_str(self): 
        spec_str = ''.join([
            self._CenQue_str(), self._Initial_str(), self._SFH_str(), self._Mass_str()
            ])
        return spec_str

    def _CenQue_str(self): 
        if self.cenque == 'default': 
            tf = 7 
            abcrun = 'RHOssfrfq_TinkerFq_Std'
            prior = 'updated'
        else: 
            raise NotImplementedError
        #cq_str = ''.join(['tf', str(tf), '.abc_', abcrun, '.prior_', prior])
        cq_str = self.cenque
        return cq_str

    def _Initial_str(self): 
        ''' Initial conditions whether it has assembly bias or not
        '''
        init_str = ''.join([
            '.Initial_', 
            self.evol_dict['initial']['assembly_bias'], 
            'AssemblyBias'
            ]) 
        return init_str 

    def _SFH_str(self): 
        '''
        '''
        if self.evol_dict['sfh']['name'] in ['constant_offset', 'no_scatter']: 
            sfh_str = ''.join(['.SFH', self.evol_dict['sfh']['name']])
        elif self.evol_dict['sfh']['name'] in ['random_step']: 
            sfh_str = ''.join([
                '.SFH', self.evol_dict['sfh']['name'],
                '_sigma', str(round(self.evol_dict['sfh']['sigma'],1)),
                '_dt', str(round(self.evol_dict['sfh']['dt_min'],2)),
                '_', str(round(self.evol_dict['sfh']['dt_max'],2))
                ])
        else: 
            raise ValueError
        return sfh_str 
    
    def _Mass_str(self): 
        mass_str = ''.join([ 
            '.mass_', self.evol_dict['mass']['type'], 
            '_tstep', str(self.evol_dict['mass']['t_step']) 
            ])
        return mass_str

    def Write(self):  
        ''' Run the evolver on the CentralMS object with the 
        evol_dict specificiations and then save to file. 
        '''
        t_start = time.time() 
        cms = CentralMS(cenque=self.cenque)
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
                't_quench', 'Minteg_hist', 'Msham_hist', 'Mhalo_hist']: 
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
    for tstep in [0.1, 0.05, 0.01]: 
        evol_dict = {
                'initial': {'assembly_bias': 'longterm', 'scatter': 0.}, 
                'sfh': {'name': 'constant_offset'}, 
                'mass': {'type': 'euler', 'f_retain': 0.6, 't_step': tstep} 
                } 
        EGP = EvolvedGalPop(cenque='default', evol_dict=evol_dict)
        EGP.Write() 
    # 'sfh': {'name': 'random_step', 'sigma':0.3, 'dt_min': 0.1, 'dt_max':0.5}, 
    #cms = CentralMS()
    #cms._Read_CenQue()

    #blah = AssignSFR0(cms)
    #eev = Evolver(blah)
    #new_blah = eev()
