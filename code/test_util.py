
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM 
from scipy import interpolate

import util as UT
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


def test_fit_zoft(): 
    z_table, t_table = UT.zt_table()
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.274)

    prettyplot()
    fig = plt.figure()
    sub = fig.add_subplot(111)

    for deg in range(2,10): 
        coeff = UT.fit_zoft(deg)
        if deg > 5: 
            print 'deg = ', deg, coeff
        zoft = np.poly1d(coeff)
        
        z_arr = np.arange(0., 2., 0.1)
        t_arr = cosmo.age(z_arr).value 

        sub.plot(t_arr, (zoft(t_arr) - z_arr)/z_arr, label='Degree='+str(deg))

    z_of_t = interpolate.interp1d(t_arr, z_arr, kind='cubic') 
    zint = z_of_t(t_table[1:20])#np.interp(t_table[:20], t_arr, z_arr)

    sub.scatter(t_table[1:20], (z_table[1:20] - zint)/zint, c='k', s=30) 
    sub.plot(np.arange(0., 14., 0.1), np.repeat(-0.025, len(np.arange(0., 14., 0.1))), c='k', ls='--', lw=3)
    sub.plot(np.arange(0., 14., 0.1), np.repeat(0.025, len(np.arange(0., 14., 0.1))), c='k', ls='--', lw=3)

    sub.set_ylim([-0.05, 0.05])
    sub.set_xlim([3, 13.8])
    sub.legend(loc='upper left')
    plt.show()



if __name__=='__main__':
    test_fit_zoft()
