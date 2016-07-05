'''



'''
import h5py
import numpy as np

# --- local --- 
import util as UT


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
        f.close() 
        return None 


def AssignSFR0(cms): 
    ''' Assign initial SFRs to the cms object based on tsnap_genesis 
    (time when the halo enters the catalog) and mass_genesis
    '''
    if 'tsnap_genesis' not in cms.__dict__.keys(): 
        # Most likely you did not read in CenQue catalog!
        raise ValueError

    # Assign SFR to star-forming galaxies 
    sfr_class[starforming] = 'star-forming'
    mu_sf_sfr = AverageLogSFR_sfms(
            mass[starforming], 
            redshift[starforming], 
            sfms_prop=sfms_dict)
    sigma_sf_sfr = ScatterLogSFR_sfms(
            mass[starforming], 
            redshift[starforming],
            sfms_prop=sfms_dict)
    avg_sfr[starforming] = mu_sf_sfr
    delta_sfr[starforming] = sigma_sf_sfr * np.random.randn(ngal_sf)
    sfr[starforming] = mu_sf_sfr + delta_sfr[starforming]
    ssfr[starforming] = sfr[starforming] - mass[starforming]





if __name__=='__main__': 
    cms = CentralMS()
    cms._Read_CenQue()
