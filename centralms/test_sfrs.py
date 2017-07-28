'''


Test functions for handling star formation rates


'''
import numpy as np
from scipy.integrate import odeint

import sfrs as SFR



def IntegrationTest(): 
    ''' Simple test the integration
    '''

    logsfr = lambda mstar, t: np.log10(t**2)
    
    for tt in np.arange(1., 11., 1.):
        M_int = SFR.integSFR(logsfr, np.array([0.]), 
                np.array([0.]), np.array([tt]),
                mass_dict={'type': 'rk4', 'f_retain': 1e-9, 't_step': 0.01})

        print np.log10(10**M_int[0] - 1.), np.log10(tt**3/3.)
    return None


def Integration_ScipyComp(): 
    ''' Simple test the integration
    '''
    dydt = lambda y, t: t
    
    M_euler = SFR.ODE_Euler(dydt, np.array([0.]), np.array([0.,10.]), 0.001)
    M_RK4 = SFR.ODE_RK4(dydt, np.array([0.]), np.array([0.,10.]), 0.1)
    M_scipy = odeint(dydt, np.array([0.]), np.array([0.,10.])) 
    print M_euler
    print M_RK4
    print M_scipy

    
    return None




if __name__=='__main__':
    #IntegrationTest()
    Integration_ScipyComp() 
