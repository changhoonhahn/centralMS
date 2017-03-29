
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM 
from scipy import interpolate
import util as UT


def test_fit_zoft(): 
    z_table, t_table = UT.zt_table()
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    fig = plt.figure()
    sub = fig.add_subplot(111)

    for deg in range(2,7): 
        coeff = UT.fit_zoft(deg)
        zoft = np.poly1d(coeff)
        
        z_arr = np.arange(0., 2., 0.1)
        t_arr = cosmo.age(z_arr).value 

        sub.plot(t_arr, (zoft(t_arr) - z_arr)/z_arr, label='Degree='+str(deg))

    z_of_t = interpolate.interp1d(t_arr, z_arr, kind='cubic') 
    print t_table[:20]
    print t_arr
    zint = z_of_t(t_table[:20])#np.interp(t_table[:20], t_arr, z_arr)
    print z_table[:20]
    print zint

    sub.scatter(t_table[:20], (z_table[:20] - zint)/zint, c='k', s=30) 

    sub.set_ylim([-0.2, 0.2])
    sub.legend(loc='upper left')
    plt.show()



if __name__=='__main__':
    test_fit_zoft()
