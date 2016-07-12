'''



'''
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
        mu_logsfr = sfrs.AverageLogSFR_sfms(self.cms.mass_genesis, self.cms.zsnap_genesis, 
                sfms_dict=self.cms.sfms_dict)
        sigma_logsfr = sfrs.ScatterLogSFR_sfms(self.cms.mass_genesis, self.cms.zsnap_genesis, 
                sfms_dict=self.cms.sfms_dict)
        self.cms.sfr_genesis = mu_logsfr + sigma_logsfr * np.random.randn(len(mu_logsfr))

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
        sfh_kwargs = {'name': self.evol_dict['sfr']['name'], 'sfms': self.cms.sfms_dict} 

        if self.evol_dict['sfr']['name'] == 'constant_offset': 
            # No assembly bias. No dutycycle
            mu_logsfr = sfrs.AverageLogSFR_sfms(
                    self.cms.mass_genesis, UT.z_from_t(self.cms.tsnap_genesis), 
                    sfms_dict=self.cms.sfms_dict)
            dsfr = self.cms.sfr_genesis - mu_logsfr

            sfh_kwargs['dsfr'] = dsfr
        # ---- 

        # identify the quenching galaxies 
        qing = np.where(
                (self.cms.t_quench != 999.) & (self.cms.t_quench > 0.) 
                )  
        t_final = np.repeat(13.1328, len(self.cms.mass_genesis))    
        t_final[qing] = self.cms.t_quench[qing]
    
        z_table, t_table = UT.zt_table()    # construct z(t) interpolation function 
        z_of_t = interpolate.interp1d(list(reversed(t_table)), list(reversed(z_table)), 
                kind='cubic') 

        # ---- 
        # integrated SFR stellar mass while on SFMS 
        # for galaxies that do not quench, this is their final mass  
        # ---- 
        earliest = self.cms.tsnap_genesis.min()
        func_kwargs = {
                't_offset': self.cms.tsnap_genesis - earliest, 
                't_final': t_final, 
                'f_retain': self.evol_dict['mass']['f_retain'], 
                'zfromt': z_of_t, 
                'sfh_kwargs': sfh_kwargs
                }
        if self.evol_dict['mass']['type'] == 'rk4': 
            f_ode = sfrs.ODE_RK4
        elif self.evol_dict['mass']['type'] == 'euler': 
            f_ode = sfrs.ODE_Euler
        integ_logM = f_ode(
                sfrs.dlogMdt_MS,                    # dy/dt
                self.cms.mass_genesis,              # logM0
                np.array([earliest, 13.1328]),      # output times  
                self.evol_dict['mass']['t_step'],   # time step
                **func_kwargs) 
        #integ_logM = odeint(sfrs.logM_integrand_MS, self.cms.mass_genesis, [earliest, 13.1328], 
        #        (t_offset, t_final, sfh_kwargs,))
    
        self.cms.mass = integ_logM[-1]                  # integrated SFR mass 
        self.cms.sfr = sfrs.LogSFR_sfms(integ_logM[-1], z_of_t(t_final), sfms_dict=sfh_kwargs) 
        # ---- 

        # quenching galaxies after they're off the SFMS
        t_final = np.repeat(13.1328, len(self.cms.mass_genesis))
        tauQ = sfrs.getTauQ(self.cms.mass[qing], tau_dict=self.cms.tau_dict)

        # calculate finall stellar mass of quenching galaxies
        func_kwargs = {
                'logSFR_Q': self.cms.sfr[qing],         # log SFR_Q
                'tau_Q': tauQ,
                't_Q': self.cms.t_quench[qing],         # quenching time 
                'f_retain': self.evol_dict['mass']['f_retain'], 
                't_offset': self.cms.t_quench[qing] - self.cms.t_quench[qing].min(), 
                't_final': t_final[qing]
                }
        integ_logM_Q = sfrs.ODE_RK4(
                sfrs.dlogMdt_Q, 
                self.cms.mass[qing], 
                np.array([self.cms.t_quench[qing].min(), 13.1328]), 
                self.evol_dict['mass']['t_step'],   # time step
                **func_kwargs)
        self.cms.mass[qing] = integ_logM_Q[-1]
        self.cms.sfr[qing] = sfrs.LogSFR_Q(
                13.1328, 
                logSFR_Q=self.cms.sfr[qing],         # log SFR_Q
                tau_Q=tauQ,
                t_Q=self.cms.t_quench[qing])
        return None


class EvolvedGalPop(GalPop): 
    def __init__(self, cenque='default', evol_dict=None): 
        ''' 
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
        if self.cenque == 'default': 
            tf = 7 
            abcrun = 'RHOssfrfq_TinkerFq_Std'
            prior = 'updated'
        else: 
            raise NotImplementedError

        # Write Star forming and Quenching catalog 
        evol_file = ''.join([
            '/data1/hahn/centralMS/galpop/', 
            'sfms.centrals.', 
            'tf', str(tf), 
            '.abc_', abcrun, 
            '.prior_', prior_name, 
            '.sfr_', self.evol_dict['sfr']['name'], 
            '.mass_', self.evol_dict['mass']['type'], 
            '_tstep', str(self.evol_dict['mass']['t_step']), 
            '.hdf5'])
        return evol_file 

    def Write(self):  
        '''
        '''
        cms = CentralMS(cenque=self.cenque)
        cms._Read_CenQue()
        eev = Evolver(cms, evol_dict=self.evol_dict)
        MSpop = eev()

        f = h5py.File(self.File(), 'w')    
        grp = f.create_group('data')
        # hardcoded columns for the catalogs
        for col in ['mass', 'sfr', 'halo_mass',
                'tsnap_genesis', 'nsnap_genesis', 'zsnap_genesis', 
                'mass_genesis', 'halomass_genesis', 
                't_quench', 'Minteg_hist', 'Msham_hist']: 
            grp.create_dataset(col, data = getattr(MSpop, col)) 
        f.close()
        return None

    def Read(self): 
        '''
        '''
        f = h5py.File(self.File(), 'r')    
        grp = f['data']

        for key in grp.keys(): 
            setattr(self, key, grp[key][:])

        f.close()
        return None






if __name__=='__main__': 

    evol_dict = {
            'sfr': {'name': 'no_scatter'}, 
            'mass': {'type': 'rk4', 'f_retain': 0.6, 't_step': 0.01} 
            } 
    EGP = EvolvedGalPop(cenque='default', evol_dict=evol_dict)
    EGP.Write() 

    #cms = CentralMS()
    #cms._Read_CenQue()

    #blah = AssignSFR0(cms)
    #eev = Evolver(blah)
    #new_blah = eev()
