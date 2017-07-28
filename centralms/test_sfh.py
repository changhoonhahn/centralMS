'''

Test methods in sfh.py 

'''
import numpy as np 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 

import sfh as SFH
import util as UT


def test_IntegratedSFH(test, solver='euler'):
    ''' Come up with some tests 
    '''

    if solver == 'euler': 
        ODEsolver = SFH.ODE_Euler
    elif solver == 'rk4':
        ODEsolver = SFH.ODE_RK4

    if test == '1':  # simplest test (PASSED)
        z_of_t = lambda t: 1. 

        logSFR_M_z = lambda mm, zz: -9.

        dlogmdz_kwargs = {
                'logsfr_M_z': logSFR_M_z, 
                'f_retain': 1., 
                'zoft': z_of_t
                }
        
        tt, output = ODEsolver(SFH.dlogMdt, np.array([0.]), np.arange(0., 10., 1.), 0.01, **dlogmdz_kwargs)

        fig = plt.figure()
        sub = fig.add_subplot(111)
        sub.plot(tt, [10**out[0] for out in output], c='b', lw=2) 
        sub.plot(np.arange(0., 10., 1.), 1.+ np.arange(0., 10., 1), c='k', ls='--', lw=3)

        plt.show()

    elif test == '2': # time dependent dlogm/dt (PASSED)
        z_of_t = lambda t: t 

        logSFR_M_z = lambda mm, zz: np.log10(zz**2)

        dlogmdz_kwargs = {
                'logsfr_M_z': logSFR_M_z, 
                'f_retain': 1.e-9, 
                'zoft': z_of_t
                }
        tt, output = ODEsolver(SFH.dlogMdt, np.array([0.]), np.arange(0., 10., 1.), 0.01, **dlogmdz_kwargs)

        fig = plt.figure()
        sub = fig.add_subplot(111)
        sub.plot(tt, [10**out[0] for out in output], c='b', lw=2) 
        sub.plot(np.arange(0., 10., 1.), 1.+ (np.arange(0., 10., 1)**3)/3., c='k', ls='--', lw=3)

        plt.show()
    
    elif test == '3': 
        z_of_t = lambda t: 1 

        logSFR_M_z = lambda mm, zz: mm

        dlogmdz_kwargs = {
                'logsfr_M_z': logSFR_M_z, 
                'f_retain': 1.e-9, 
                'zoft': z_of_t
                }
        tt, output = ODEsolver(SFH.dlogMdt, np.array([0.]), np.arange(0., 10., 1.), 0.01, **dlogmdz_kwargs)

        fig = plt.figure()
        sub = fig.add_subplot(111)
        sub.plot(tt, [10**out[0] for out in output], c='b', lw=2) 
        sub.plot(np.arange(0., 10., 1.), np.exp(np.arange(0., 10., 1)), c='k', ls='--', lw=3)

        plt.show()


def test_zt_interpolate(): 
    z_table, t_table = UT.zt_table()     
    z_of_t1 = interp1d(list(reversed(t_table)), list(reversed(z_table)), kind='cubic') 
    z_of_t2 = interp1d(t_table, z_table, kind='cubic') 
    z_of_t3 = interp1d(list(reversed(t_table[:25])), list(reversed(z_table[:25])), kind='cubic') 
    z_of_t4 = interp1d(t_table[:25], z_table[:25], kind='cubic') 

    fig = plt.figure()
    sub = fig.add_subplot(111)

    t_arr = np.arange(t_table[24], t_table[0], 0.1)
    sub.plot(t_table[:20], (z_table[:20] - z_of_t1(t_table[:20]))/z_table[:20])
    sub.plot(t_table[:20], (z_table[:20] - z_of_t2(t_table[:20]))/z_table[:20])
    sub.plot(t_table[:20], (z_table[:20] - z_of_t3(t_table[:20]))/z_table[:20])
    sub.plot(t_table[:20], (z_table[:20] - z_of_t4(t_table[:20]))/z_table[:20])
    
    #sub.scatter(t_table[:20], z_table[:20], color='k', s=10, lw=0)
    plt.show()




if __name__=='__main__': 
    test_zt_interpolate()
    #test_IntegratedSFH('2', solver='rk4')

