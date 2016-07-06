'''



'''
import h5py
import numpy as np

# --- local --- 
import util as UT
import sfrs 


class CentralMS(object):

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
        if self.cenque == 'default': 
            tf = 7 
            abcrun = 'RHOssfrfq_TinkerFq_Std'
            prior = 'updated'
        else: 
            raise NotImplementedError

        file = ''.join([UT.dat_dir(), 'cenque/',
            'sfms.centrals.', 
            'tf', str(tf), 
            '.abc_', abcrun, 
            '.prior_', prior, 
            '.hdf5']) 

        # read in the file and save to object
        f = h5py.File(file, 'r')  
        grp = f['data'] 
        for col in grp.keys(): 
            if col == 'mass': 
                # make sure to mark as SHAM mass
                setattr(self, 'M_sham', grp[col][:])    
            elif col in ['sfr', 'ssfr']:
                continue 
            else: 
                setattr(self, col, grp[col][:])

        for key in grp.attrs.keys(): 
            if key in ['sfms', 'tau']: 
                attr_dict = {}
                for keyind in grp.attrs[key].split(','): 
                    try: 
                        attr_dict[keyind.split(':')[0]] = float(keyind.split(':')[1])
                    except ValueError:
                        attr_dict[keyind.split(':')[0]] = keyind.split(':')[1]

                if key == 'tau': 
                    attr_dict['name'] = 'line'

                setattr(self, key+'_dict', attr_dict)
            else:
                setattr(self, key+'_attr', grp.attrs[key])

        f.close() 
            
        # some small pre-processing here so convenience 
        setattr(self, 'zsnap_genesis', UT.z_from_t(self.tsnap_genesis)) 

        return None 


def AssignSFR0(cms): 
    ''' Assign initial SFRs to the cms object based on tsnap_genesis 
    (time when the halo enters the catalog) and mass_genesis
    '''
    if 'tsnap_genesis' not in cms.__dict__.keys(): 
        # Most likely you did not read in CenQue catalog!
        raise ValueError

    # Assign SFR to star-forming galaxies 
    mu_logsfr = sfrs.AverageLogSFR_sfms(cms.mass_genesis, cms.zsnap_genesis, 
            sfms_dict=cms.sfms_dict)
    sigma_logsfr = sfrs.ScatterLogSFR_sfms(cms.mass_genesis, cms.zsnap_genesis, 
            sfms_dict=cms.sfms_dict)
    cms.sfr_genesis = mu_logsfr + sigma_logsfr

    return cms 



class Evolver(object): 
    
    def __init__(self, cms, evol_dict=None): 
        ''' Class object that evolves the CentralMS galaxy catalog catalog object .
        Object contains suite of functions for the evolution. 
        '''
        self.cms = cms 
        if evol_dict is None:  # default 
            self.evol_dict = {
                    'sfr': {'name': 'nothing'}, 
                    'mass': {'type': 'rk4', 'f_retain': 0.6, 't_step': 0.1} 
                    }
        else: 
            self.evol_dict = evol_dict
        self.Evolve()

    def __call__(self):  
        return self.cms 

    def Evolve(self, evol_dict=None): 
        ''' Evolve SFR and calculated integrated SFR stellar masses. 
        The module creates lambda functions for log SFR(t) and then 
        integrates that. Currently set up to minimize the amount of
        specific cases. 
        '''
        # Construct functions of SFR(t) 
        sfr_dict = self.evol_dict['sfr']
        if sfr_dict['name'] == 'nothing':
            # No assembly bias. No dutycycle
            mu_logsfr = sfrs.AverageLogSFR_sfms(
                    self.cms.mass_genesis, UT.z_from_t(self.cms.tsnap_genesis), 
                    sfms_dict=self.cms.sfms_dict)
            dsfr = self.cms.sfr_genesis - mu_logsfr

            self.logSFRt_MS = lambda mstar, t: sfrs.SFRt_MS_nothing(mstar, t, dsfr, 
                    sfms_dict=self.cms.sfms_dict) 
        else: 
            raise NotImplementedError
    
        # Now integrate to get stellar mass 
        mass_dict = self.evol_dict['mass']

        qing = np.where(self.cms.t_quench != 999.)  # quenching galaxies 
    
        t_final = np.repeat(13.1328, len(self.cms.mass_genesis))
        t_final[qing] = self.cms.t_quench[qing]
    
        # integrated SFR stellar mass while on SFMS 
        # for galaxies that do not quench, this is their final mass  
        integ_mass, sfr_f = sfrs.integSFR(self.logSFRt_MS, self.cms.mass_genesis, 
                self.cms.tsnap_genesis, t_final, mass_dict=mass_dict) 
    
        self.cms.sfr = sfr_f 
        self.cms.mass = integ_mass
        t_final = np.repeat(13.1328, len(self.cms.mass_genesis))

        # quenching galaxies after they're off the SFMS
        if sfr_dict['name'] == 'nothing':
            self.logSFRt_Q = lambda mstar, t: sfrs.SFRt_Q_nothing(
                    self.cms.mass[qing], t, dsfr[qing], 
                    sfms_dict=self.cms.sfms_dict, tau_dict=self.cms.tau_dict)
        else: 
            raise NotImplementedError
    
        # calculate finall stellar mass of quenching galaxies
        integ_mass_Q, sfrQ_f= sfrs.integSFR(self.logSFRt_Q, self.cms.mass[qing], 
                self.cms.t_quench[qing], t_final[qing], mass_dict=mass_dict) 
        self.cms.mass[qing] = integ_mass_Q
        self.cms.sfr[qing] = sfrQ_f 

        return None




if __name__=='__main__': 
    cms = CentralMS()
    cms._Read_CenQue()

    blah = AssignSFR0(cms)
    eev = Evolver(blah)
    new_blah = eev()
