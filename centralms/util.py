'''

General utility functions 

'''
import os
import sys
import numpy as np
from scipy import interpolate
from astropy import units as U
from astropy import constants as Const
from astropy.cosmology import WMAP7


def check_env(): 
    if os.environ.get('CENTRALMS_DIR') is None: 
        raise ValueError("set $CENTRALMS_DIR in bashrc file!") 
    return None


def dat_dir(): 
    '''
    '''
    return os.environ.get('CENTRALMS_DIR') 


def code_dir(): 
    ''' Directory where all the code is located (the directory that this file is in!)
    '''
    return os.environ.get('CENTRALMS_CODEDIR') 


def fig_dir(): 
    return ''.join([code_dir(), 'fig/']) 


def doc_dir(): 
    return ''.join([code_dir(), 'doc/']) 


def tex_dir(): 
    return ''.join([code_dir(), 'doc/']) 


def bar_plot(bin_edges, values): 
    ''' Take outputs from numpy histogram and return pretty bar plot
    '''
    xx = [] 
    yy = [] 

    for i_val, val in enumerate(values): 
        xx.append(bin_edges[i_val]) 
        yy.append(val)
        xx.append(bin_edges[i_val+1]) 
        yy.append(val)

    return [np.array(xx), np.array(yy)]


def median(data, weights=None): 
    ''' Median in case there's weights 
    '''
    if weights is None: 
        return np.median(datas)
    else:
        # below is taken from 
        # www.github.com/tinybike/weightedstats
        midpoint = 0.5 * sum(weights)
        
        if any([j > midpoint for j in weights]):
            return data[weights.index(max(weights))]

        if any([j > 0 for j in weights]):
            sorted_data, sorted_weights = zip(*sorted(zip(data, weights)))
            cumulative_weight = 0
            below_midpoint_index = 0
            while cumulative_weight <= midpoint:
                below_midpoint_index += 1
                cumulative_weight += sorted_weights[below_midpoint_index-1]
            cumulative_weight -= sorted_weights[below_midpoint_index-1]
            if cumulative_weight - midpoint < sys.float_info.epsilon:
                bounds = sorted_data[below_midpoint_index-2:below_midpoint_index]
                return sum(bounds) / float(len(bounds))

            return sorted_data[below_midpoint_index-1]


def tdyn_t(tt, deg=6): 
    if deg == 6: 
        coeff = np.array([ 9.29206984e-09, -3.11237220e-07, 6.89619438e-06, 1.49737134e-04, 2.60164944e-04, 1.49149975e-01, 4.31546682e-04])
    else: raise NotImplementedError 
    tdyn_t = np.poly1d(coeff)
    return tdyn_t(tt)


def tdyn_nsnap(nsnap): 
    z_t = z_nsnap(nsnap)
    rho_m = WMAP7.Om(z_t) * WMAP7.critical_density(z_t)
    return ((4./3. * np.pi * 200. * rho_m * Const.G)**-0.5).to(U.Gyr).value


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


def fit_zoft(deg): 
    z_table, t_table = zt_table()

    p = np.polyfit(t_table[:25], z_table[:25], deg) 
    return p


def z_of_t(tcosmic, deg=6): 
    if deg == 6: 
        coeff = [4.79981405e-06, -2.78012322e-04, 6.67694215e-03, -8.61053273e-02, 6.45836541e-01, -2.88397725e+00, 6.93399197e+00]
    elif deg == 7: 
        coeff = [-6.25747556e-07, 4.13664580e-05, -1.16240613e-03, 1.81118817e-02, -1.71157185e-01, 1.00848397e+00, -3.70236957e+00, 7.68673323e+00]
    else:
        raise NotImplementedError
    zt = np.poly1d(coeff)
    return zt(tcosmic)


def fit_tofz(deg): 
    z_table, t_table = zt_table()

    p = np.polyfit(z_table[:25], t_table[:25], deg) 
    return p


def t_of_z(red, deg=6): 
    if deg == 6: 
        coeff = [0.04018147, -0.39686453, 1.77280041, -4.899108, 9.59869045, -13.93909509, 13.80933101]
    elif deg == 7: 
        coeff = [-0.01410059, 0.14902284, -0.72664217, 2.2683614, -5.28231397, 9.74089113, -13.95932603, 13.80984062]
    else:
        raise NotImplementedError
    tz = np.poly1d(coeff)
    return tz(red)


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


def dt_dz(zz, t_of_z=None): 
    # dt/dz estimate from Hogg 1999 per Gyr because it's faster
    if zz < 1.:  
        return -13.8835 + 19.3598 * zz - 13.621 * zz**2 + 4.2141 * zz**3
    else: 
        if t_of_z is not None: 
            return (t_of_z(zz + 0.01) - t_of_z(zz - 0.01))/ 0.02
        else: 
            return (t_from_z(zz + 0.01) - t_from_z(zz - 0.01))/ 0.02
    

def zt_table(): 
    ''' Return z and t tables in case you want to construct the interpolation function
    within another function
    '''
    z = np.array([0.0000, 0.0502, 0.1028, 0.1581, 0.2162, 0.2771, 0.3412, 0.4085, 0.4793, 0.5533, 0.6313, 0.7132, 0.7989, 0.8893, 0.9841, 1.0833, 1.1882, 1.2978, 1.4131, 1.5342, 1.6610, 1.7949, 1.9343, 2.0817, 2.2362, 2.3990, 2.5689, 2.7481, 2.9370, 3.1339, 3.3403, 3.5579, 3.7870])
    t = np.array([13.8099, 13.1328, 12.4724, 11.8271, 11.1980, 10.5893, 9.9988, 9.4289, 8.8783, 8.3525, 7.8464, 7.3635, 6.9048, 6.4665, 6.0513, 5.6597, 5.2873, 4.9378, 4.6080, 4.2980, 4.0079, 3.7343, 3.4802, 3.2408, 3.0172, 2.8078, 2.6136, 2.4315, 2.2611, 2.1035, 1.9569, 1.8198, 1.6918])

    return [z,t]


def z_nsnap(nsnap): 
    # given n_snapshot get redshift
    z = np.array([0.0000, 0.0502, 0.1028, 0.1581, 0.2162, 0.2771, 0.3412, 0.4085, 0.4793, 0.5533, 0.6313, 0.7132, 0.7989, 0.8893, 0.9841, 1.0833, 1.1882, 1.2978, 1.4131, 1.5342, 1.6610, 1.7949, 1.9343, 2.0817, 2.2362, 2.3990, 2.5689, 2.7481, 2.9370, 3.1339, 3.3403, 3.5579, 3.7870])
    return z[nsnap]


def t_nsnap(nsnap): 
    # given n_snapshot return t_cosmic 
    t = np.array([13.8099, 13.1328, 12.4724, 11.8271, 11.1980, 10.5893, 9.9988, 9.4289, 8.8783, 8.3525, 7.8464, 7.3635, 6.9048, 6.4665, 6.0513, 5.6597, 5.2873, 4.9378, 4.6080, 4.2980, 4.0079, 3.7343, 3.4802, 3.2408, 3.0172, 2.8078, 2.6136, 2.4315, 2.2611, 2.1035, 1.9569, 1.8198, 1.6918])
    return t[nsnap]


def replicate(arr, n): 
    ''' Given array or value, produce a "blank" numpy array of length n 
    that has the same variable type as arr. 
    '''
    try: 
        if isinstance(arr, str): 
            val = arr
        else: 
            val = arr[0]
    except TypeError: 
        val = arr

    if isinstance(val, int): 
        return np.tile(-999, n)
    elif isinstance(val, float) or isinstance(val, np.float32) or isinstance(val, np.float64): 
        return np.tile(-999., n)
    elif isinstance(val, str): 
        if len(val) < 16: 
            return np.tile(val, n).astype('|S16')
        else: 
            leng = str(np.ceil(len(val)/16)*16)
            return np.tile(val, n).astype('|S'+leng)
    else: 
        print val, type(val) 
        raise ValueError


def weighted_quantile(values, quantiles, weights=None, values_sorted=False, old_style=False):
    """ 
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param weights: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if weights is None:
        weights = np.ones(len(values))
    weights = np.array(weights)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]' 
    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]

    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    if old_style: 
        # To be convenient with np.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(weights)
    return np.interp(quantiles, weighted_quantiles, values)


def GrabLocalFile(string, machine='harmattan'): 
    ''' scp dat_dir()+string from machine (harmattan)
    '''
    # parse subdirectory
    sub_dir = '/'.join(string.split('/')[:-1] + [''])

    if machine == 'harmattan': 
        scp_cmd = "scp harmattan:/data1/hahn/centralMS/"+string+" "+dat_dir()+sub_dir
        print scp_cmd
        os.system(scp_cmd)
    else: 
        raise NotImplementedError

    return None
