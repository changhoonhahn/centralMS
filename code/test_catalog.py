'''




'''
import numpy as np 

import catalog as Cat
import observables as Obvs

import matplotlib.pyplot as plt 


def plotPureCentral_SHMF(nsnap_ancestor=20):
    ''' Plot the 
    '''
    subhist = Cat.PureCentralHistory(nsnap_ancestor=20)
    subcat = subhist.Read(downsampled='33')
    
    fig = plt.figure(1)
    sub = fig.add_subplot(111)

    for i in range(1, 21): 
        if i == 1: 
            masses = subcat['halo.m']
            weights = subcat['weights']
        else: 
            masses = subcat['snapshot'+str(i)+'_halo.m']
            weights = subcat['weights']

        shmf = Obvs.getMF(masses, weights=weights, m_arr=np.arange(10., 15.5, 0.1))
        sub.plot(shmf[0], shmf[1]) 

    # x-axis
    sub.set_xlim([10., 15.])
    # y-axis
    sub.set_yscale("log") 
    plt.show() 





if __name__=='__main__': 
    plotPureCentral_SHMF(nsnap_ancestor=20)

    #subhist = Cat.SubhaloHistory(nsnap_ancestor=20)
    #subhist._CheckHistory()
    #subhist = Cat.PureCentralHistory(nsnap_ancestor=20)
    #subhist.Build()
    #subhist.Downsample()
