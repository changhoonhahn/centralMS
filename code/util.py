'''

General utility functions 

'''
import os
import numpy as np
from scipy import interpolate


def code_dir(): 
    ''' Directory where all the code is located (the directory that this file is in!)
    '''
    return os.path.dirname(os.path.realpath(__file__))


def dat_dir(): 
    ''' dat directory is symlinked to a local path where the data files are located
    '''
    return os.path.dirname(os.path.realpath(__file__)).split('code')[0]+'dat/'


def z_from_t(tcosmic): 
    ''' Given cosmic time return redshift using spline interpolation of snapshot table 

    Parameters
    ----------
    tcosmic : cosmic time 

    Notes
    -----
    * Only worry is that it may take too long
    '''
    # read in snapshot table file 
    z = np.array([0.0000, 0.0502, 0.1028, 0.1581, 0.2162, 0.2771, 0.3412, 0.4085, 0.4793, 0.5533, 0.6313, 0.7132, 0.7989, 0.8893, 0.9841, 1.0833, 1.1882, 1.2978, 1.4131, 1.5342, 1.6610, 1.7949, 1.9343, 2.0817, 2.2362, 2.3990, 2.5689, 2.7481, 2.9370, 3.1339, 3.3403, 3.5579, 3.7870])
    t = np.array([13.8099, 13.1328, 12.4724, 11.8271, 11.1980, 10.5893, 9.9988, 9.4289, 8.8783, 8.3525, 7.8464, 7.3635, 6.9048, 6.4665, 6.0513, 5.6597, 5.2873, 4.9378, 4.6080, 4.2980, 4.0079, 3.7343, 3.4802, 3.2408, 3.0172, 2.8078, 2.6136, 2.4315, 2.2611, 2.1035, 1.9569, 1.8198, 1.6918])

    z_of_t = interpolate.interp1d(list(reversed(t)), list(reversed(z)), kind='cubic') 

    return z_of_t(tcosmic) 


def t_from_z(redshift): 
    ''' Given redshift, return cosmic time using spline interpolation of snapshot table

    Parameters
    ----------
    redshift : redshift 

    Notes
    -----
    * Only worry is that it may take too long

    '''
    # read in snapshot table file 
    z = np.array([0.0000, 0.0502, 0.1028, 0.1581, 0.2162, 0.2771, 0.3412, 0.4085, 0.4793, 0.5533, 0.6313, 0.7132, 0.7989, 0.8893, 0.9841, 1.0833, 1.1882, 1.2978, 1.4131, 1.5342, 1.6610, 1.7949, 1.9343, 2.0817, 2.2362, 2.3990, 2.5689, 2.7481, 2.9370, 3.1339, 3.3403, 3.5579, 3.7870])
    t = np.array([13.8099, 13.1328, 12.4724, 11.8271, 11.1980, 10.5893, 9.9988, 9.4289, 8.8783, 8.3525, 7.8464, 7.3635, 6.9048, 6.4665, 6.0513, 5.6597, 5.2873, 4.9378, 4.6080, 4.2980, 4.0079, 3.7343, 3.4802, 3.2408, 3.0172, 2.8078, 2.6136, 2.4315, 2.2611, 2.1035, 1.9569, 1.8198, 1.6918])

    t_of_z = interpolate.interp1d(z, t, kind='cubic') 

    return t_of_z(redshift) 
