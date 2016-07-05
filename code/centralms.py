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
            setattr(self, key+'_attr', grp.attrs[key])

        f.close() 
        return None 


def AssignSFR0(cms): 
    ''' Assign initial SFRs to the cms object based on tsnap_genesis 
    (time when the halo enters the catalog) and mass_genesis
    '''
    if 'tsnap_genesis' not in cms.__dict__.keys(): 
        # Most likely you did not read in CenQue catalog!
        raise ValueError
    
    z_genesis = UT.z_from_t(cms.tsnap_genesis) 

    # Assign SFR to star-forming galaxies 
    sfms_dict = {}
    for keyind in cms.sfms_attr.split(','): 
        try: 
            sfms_dict[keyind.split(':')[0]] = float(keyind.split(':')[1])
        except ValueError:
            sfms_dict[keyind.split(':')[0]] = keyind.split(':')[1]

    mu_logsfr = sfrs.AverageLogSFR_sfms(cms.mass_genesis, z_genesis, sfms_dict=sfms_dict)
    sigma_logsfr = sfrs.ScatterLogSFR_sfms(cms.mass_genesis, z_genesis, sfms_dict=sfms_dict)

    sfr_genesis = mu_logsfr + sigma_logsfr
    cms.sfr_genesis = sfr_genesis

    return cms 


if __name__=='__main__': 
    cms = CentralMS()
    cms._Read_CenQue()

    blah = AssignSFR0(cms)
    print blah.sfr_genesis
